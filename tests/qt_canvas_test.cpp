#include "EdgeItem.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "SubflowItem.hpp"

#include "application/application.h"
#include "model/model.h"

#include <QTemporaryDir>
#include <QGraphicsRectItem>
#include <QImage>
#include <QPainter>
#include <QFontMetricsF>
#include <QLineF>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QUuid>
#include <QWheelEvent>
#include <QScrollBar>
#include <QtTest>

#include <memory>
#include <cstring>

class QtCanvasTest final : public QObject {
    Q_OBJECT
private:
    std::unique_ptr<QTemporaryDir> directory_;
    QByteArray projectPath_;
    NNApplication *application_ = nullptr;
    GraphScene *scene_ = nullptr;
    GraphView *view_ = nullptr;

    bool addNode(const char *id, const char *package, const char *scope,
                 double x, double y) {
        char error[512] = {};
        return nn_app_add_node(application_, id, package, "0.1.0", scope, x, y,
                               error, sizeof(error));
    }

    void pumpEvents() {
        QCoreApplication::processEvents();
        QTest::qWait(30);
        QCoreApplication::processEvents();
    }

    QImage renderPort(PortItem *port) {
        QImage image(15, 15, QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        QPainter painter(&image);
        painter.translate(7.0, 7.0);
        port->paint(&painter, nullptr);
        painter.end();
        return image;
    }

private slots:
    void init() {
        directory_ = std::make_unique<QTemporaryDir>();
        QVERIFY(directory_->isValid());
        const QByteArray core = QByteArray(NN_SOURCE_DIR) + "/stereotype-packages/core";
        application_ = nn_app_new(core.constData());
        QVERIFY(application_);
        const QByteArray parent = directory_->path().toUtf8();
        const QByteArray projectId = (QStringLiteral("canvas-") +
            QUuid::createUuid().toString(QUuid::WithoutBraces)).toUtf8();
        projectPath_ = (directory_->path() + QLatin1Char('/') + QString::fromUtf8(projectId)).toUtf8();
        char error[512] = {};
        QVERIFY2(nn_app_create(application_, parent.constData(), projectId.constData(), "Canvas", false,
                               error, sizeof(error)), error);
        QVERIFY(addNode("source", "core.input", "", 100, 100));
        QVERIFY(addNode("target", "core.relu", "", 100, 280));
        QVERIFY(addNode("flow", "core.subflow-proxy", "", 420, 100));
        QVERIFY(addNode("child", "core.relu", "flow", 100, 100));
        scene_ = new GraphScene(application_);
        view_ = new GraphView(scene_);
        view_->resize(900, 600);
        view_->show();
        scene_->refresh();
        pumpEvents();
    }

    void cleanup() {
        delete view_;
        view_ = nullptr;
        delete scene_;
        scene_ = nullptr;
        nn_app_free(application_);
        application_ = nullptr;
    }

    void movementPersistsOnRelease() {
        NodeItem *node = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(node);
        const QPoint start = view_->mapFromScene(node->sceneBoundingRect().center());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), start + QPoint(38, 27), 30);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier,
                            start + QPoint(38, 27));
        pumpEvents();
        const NNNode *saved = nn_model_find_node(nn_app_model(application_), "target");
        QVERIFY(saved);
        QCOMPARE(saved->x, 140.0);
        QCOMPARE(saved->y, 300.0);
    }

    void portsConnectThroughApplication() {
        NodeItem *source = scene_->nodeItem(QStringLiteral("source"));
        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(source && target);
        PortItem *out = nullptr;
        PortItem *in = nullptr;
        for (PortItem *port : source->ports())
            if (port->isOutput() && port->handleId() == QStringLiteral("out")) out = port;
        for (PortItem *port : target->ports())
            if (!port->isOutput() && port->handleId() == QStringLiteral("in")) in = port;
        QVERIFY(out && in);
        const QPoint start = view_->mapFromScene(out->scenePos());
        const QPoint end = view_->mapFromScene(in->scenePos());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), end, 30);
        QGraphicsPathItem *draft = nullptr;
        for (QGraphicsItem *item : scene_->items()) {
            if (dynamic_cast<EdgeItem *>(item)) continue;
            if (auto *path = dynamic_cast<QGraphicsPathItem *>(item)) draft = path;
        }
        QVERIFY(draft);
        QVERIFY(draft->path().elementAt(1).y > draft->path().elementAt(0).y);
        QVERIFY(qAbs(draft->path().elementAt(1).x - draft->path().elementAt(0).x) < 0.1);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier, end);
        pumpEvents();
        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(1));
        QCOMPARE(scene_->edgeItem(QStringLiteral("edge")) == nullptr, true);
        const NNEdge *edge = nn_model_edge_at(nn_app_model(application_), 0);
        QVERIFY(edge);
        QCOMPARE(QString::fromUtf8(edge->source_id), QStringLiteral("source"));
        QCOMPARE(QString::fromUtf8(edge->target_id), QStringLiteral("target"));
        QVERIFY(scene_->edgeItem(QString::fromUtf8(edge->id)));
    }

    void boundaryOutputAcceptsRealMouseDrag() {
        NodeItem *source = scene_->nodeItem(QStringLiteral("source"));
        QVERIFY(source);
        PortItem *out = nullptr;
        for (PortItem *port : source->ports())
            if (port->isOutput()) out = port;

        NodeItem *terminal = nullptr;
        const NNModel *model = nn_app_model(application_);
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (node && node->package_id && std::strcmp(node->package_id, "core.output") == 0 &&
                (!node->scope_id || !*node->scope_id))
                terminal = scene_->nodeItem(QString::fromUtf8(node->id));
        }
        QVERIFY(out && terminal);
        PortItem *in = nullptr;
        for (PortItem *port : terminal->ports())
            if (!port->isOutput()) in = port;
        QVERIFY(in);
        const QString terminalId = terminal->id();

        const QPoint start = view_->mapFromScene(out->scenePos());
        const QPoint end = view_->mapFromScene(in->scenePos());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), end, 30);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier, end);
        pumpEvents();

        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(1));
        const NNEdge *edge = nn_model_edge_at(nn_app_model(application_), 0);
        QVERIFY(edge);
        QCOMPARE(QString::fromUtf8(edge->source_id), QStringLiteral("source"));
        QCOMPARE(QString::fromUtf8(edge->target_id), terminalId);
    }

    void selectionDeleteAndScopeNavigation() {
        NodeItem *flow = scene_->nodeItem(QStringLiteral("flow"));
        QVERIFY(dynamic_cast<SubflowItem *>(flow));
        QCOMPARE(flow->boundingRect().height(), 135.0);
        PortItem *flowOutput = nullptr;
        for (PortItem *port : flow->ports()) if (port->isOutput()) flowOutput = port;
        QVERIFY(flowOutput);
        QCOMPARE(flowOutput->y(), flow->boundingRect().height() + 1.0 - 9.0);
        QVERIFY(scene_->nodeItem(QStringLiteral("child")) == nullptr);
        const NNModel *model = nn_app_model(application_);
        const NNNode *spawnedBoundary = nullptr;
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *candidate = nn_model_node_at(model, i);
            if (candidate && candidate->scope_id && std::strcmp(candidate->scope_id, "flow") == 0 &&
                candidate->package_id && std::strcmp(candidate->package_id, "core.output") == 0)
                spawnedBoundary = candidate;
        }
        QVERIFY(spawnedBoundary);
        QVERIFY(spawnedBoundary->boundary_handle_id);
        QCOMPARE(QString::fromUtf8(spawnedBoundary->boundary_handle_id), QStringLiteral("out"));
        scene_->setFlowDirection(FlowDirection::Horizontal);
        QSignalSpy scopeSpy(scene_, &GraphScene::scopeChanged);
        const QPoint center = view_->mapFromScene(flow->sceneBoundingRect().center());
        QTest::mouseDClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, center);
        pumpEvents();
        QCOMPARE(scene_->scope(), QStringLiteral("flow"));
        QCOMPARE(scopeSpy.count(), 1);
        QVERIFY(scene_->flowDirection() == FlowDirection::Horizontal);
        QVERIFY(scene_->nodeItem(QStringLiteral("child")));
        scene_->goToParentScope();
        pumpEvents();
        QVERIFY(scene_->scope().isEmpty());
        QVERIFY(scene_->flowDirection() == FlowDirection::Horizontal);

        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(target);
        target->setSelected(true);
        QCOMPARE(scene_->selectedNodeId(), QStringLiteral("target"));
        scene_->deleteSelection();
        pumpEvents();
        QVERIFY(nn_model_find_node(nn_app_model(application_), "target") == nullptr);
    }

    void expandedSubflowIsTranslatedReadOnlyAndDoesNotChangeModel() {
        char error[512] = {};
        QVERIFY2(addNode("child-two", "core.relu", "flow", 300, 100), "Could not add preview child");
        QVERIFY2(nn_app_connect(application_, "preview-edge", "child", "out", "child-two", "in",
                                error, sizeof(error)), error);
        scene_->refresh();
        const NNNode *modelChild = nn_model_find_node(nn_app_model(application_), "child");
        QVERIFY(modelChild);
        const QPointF savedPosition(modelChild->x, modelChild->y);
        const NNNode *modelChildTwo = nn_model_find_node(nn_app_model(application_), "child-two");
        QVERIFY(modelChildTwo);
        const QPointF savedPositionTwo(modelChildTwo->x, modelChildTwo->y);
        NodeItem *owner = scene_->nodeItem(QStringLiteral("flow"));
        QVERIFY(dynamic_cast<SubflowItem *>(owner));
        QVERIFY(!scene_->nodeItem(QStringLiteral("child")));

        auto *flowItem = dynamic_cast<SubflowItem *>(owner);
        const QPoint expand = view_->mapFromScene(flowItem->mapToScene(
            QRectF(10, flowItem->boundingRect().height() - 30,
                  flowItem->boundingRect().width() - 20, 21).center()));
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, expand);
        pumpEvents();
        owner = scene_->nodeItem(QStringLiteral("flow"));
        NodeItem *preview = scene_->nodeItem(QStringLiteral("child"));
        QVERIFY(owner && preview);
        QVERIFY(preview->isReadOnlyPreview());
        NodeItem *previewTwo = scene_->nodeItem(QStringLiteral("child-two"));
        QVERIFY(previewTwo && previewTwo->isReadOnlyPreview());
        QCOMPARE(preview->scale(), previewTwo->scale());
        QVERIFY(!(preview->flags() & QGraphicsItem::ItemIsMovable));
        QVERIFY(!(preview->flags() & QGraphicsItem::ItemIsSelectable));
        for (PortItem *port : preview->ports()) QCOMPARE(port->acceptedMouseButtons(), Qt::NoButton);
        QCOMPARE(preview->parentItem(), owner);
        QVERIFY(preview->scenePos() != savedPosition);
        auto *expandedOwner = dynamic_cast<SubflowItem *>(owner);
        QVERIFY(!expandedOwner->expandedPreviewRect().isEmpty());
        QVERIFY(preview->mapRectToParent(preview->boundingRect()).top() >
                expandedOwner->NodeItem::boundingRect().bottom());
        const QRectF frame = expandedOwner->expandedPreviewRect();
        QVERIFY(frame.width() <= 640.0 && frame.height() <= 560.0);
        QCOMPARE(qAbs(frame.center().x() - owner->boundingRect().center().x()) < 2.0, true);
        EdgeItem *previewEdge = scene_->edgeItem(QStringLiteral("preview-edge"));
        QVERIFY(previewEdge && !previewEdge->routePoints().isEmpty());
        for (const QPointF &point : previewEdge->routePoints())
            QVERIFY(frame.contains(owner->mapFromScene(point)));
        QVERIFY(!(previewEdge->flags() & QGraphicsItem::ItemIsSelectable));
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child")->x, savedPosition.x());
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child")->y, savedPosition.y());
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child-two")->x, savedPositionTwo.x());
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child-two")->y, savedPositionTwo.y());
        QCOMPARE(scene_->scope(), QString());

        scene_->toggleExpanded(QStringLiteral("flow"));
        pumpEvents();
        QVERIFY(!scene_->nodeItem(QStringLiteral("child")));
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child")->x, savedPosition.x());
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child")->y, savedPosition.y());
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child-two")->x, savedPositionTwo.x());
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "child-two")->y, savedPositionTwo.y());
    }

    void scaledPreviewKeepsInternalRoutesOpen() {
        char error[512] = {};
        QByteArray previous("child");
        for (int i = 0; i < 12; ++i) {
            const QByteArray id = QByteArray("chain-") + QByteArray::number(i);
            const QByteArray edge = QByteArray("chain-edge-") + QByteArray::number(i);
            QVERIFY(addNode(id.constData(), "core.relu", "flow", 100, 300 + i * 180));
            QVERIFY2(nn_app_connect(application_, edge.constData(), previous.constData(), "out",
                                    id.constData(), "in", error, sizeof(error)), error);
            previous = id;
        }
        scene_->toggleExpanded(QStringLiteral("flow"));
        pumpEvents();
        QVERIFY(scene_->nodeItem(QStringLiteral("child"))->scale() < 0.2);
        for (int i = 0; i < 12; ++i) {
            EdgeItem *edge = scene_->edgeItem(QStringLiteral("chain-edge-%1").arg(i));
            QVERIFY(edge && !edge->routePoints().isEmpty());
            for (int p = 1; p < edge->routePoints().size(); ++p) {
                const QPointF a = edge->routePoints()[p - 1], b = edge->routePoints()[p];
                QVERIFY(qFuzzyCompare(a.x() + 1, b.x() + 1) || qFuzzyCompare(a.y() + 1, b.y() + 1));
            }
        }
    }

    void expansionSpacesPeersWithoutPersistingOffsets() {
        char error[512] = {};
        QVERIFY2(nn_app_move_node(application_, "target", 420, 280, error, sizeof(error)), error);
        QVERIFY2(nn_app_connect(application_, "root-edge", "flow", "out", "target", "in",
                                error, sizeof(error)), error);
        scene_->refresh();
        const QPointF compact = scene_->nodeItem(QStringLiteral("target"))->pos();
        scene_->toggleExpanded(QStringLiteral("flow"));
        pumpEvents();
        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        const QPointF expanded = target->pos();
        QVERIFY(expanded.y() > compact.y());
        QVERIFY(!target->sceneBoundingRect().intersects(
            scene_->nodeItem(QStringLiteral("flow"))->sceneBoundingRect()));
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "target")->y, compact.y());
        QVERIFY(!scene_->edgeItem(QStringLiteral("root-edge"))->routePoints().isEmpty());
        view_->fitGraph();
        QCOMPARE(target->pos(), expanded);
        const QPoint start = view_->mapFromScene(target->sceneBoundingRect().center());
        const QPoint end = view_->mapFromScene(target->sceneBoundingRect().center() + QPointF(40, 0));
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), end, 30);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier, end);
        pumpEvents();
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "target")->x, compact.x() + 40);
        QCOMPARE(nn_model_find_node(nn_app_model(application_), "target")->y, compact.y());
        QVERIFY2(nn_app_undo(application_, error, sizeof(error)), error);
        scene_->refresh();
        QCOMPARE(scene_->nodeItem(QStringLiteral("target"))->pos(), expanded);
        scene_->toggleExpanded(QStringLiteral("flow"));
        pumpEvents();
        QCOMPARE(scene_->nodeItem(QStringLiteral("target"))->pos(), compact);
    }

    void emptyCanvasBackgroundIncludesVisibleGrid() {
        QImage image(240, 180, QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        QPainter painter(&image);
        scene_->render(&painter, QRectF(0, 0, 240, 180), QRectF(2000, 2000, 240, 180));
        painter.end();
        const QColor background = image.pixelColor(3, 3);
        bool foundGrid = false;
        for (int y = 0; y < image.height() && !foundGrid; ++y)
            for (int x = 0; x < image.width(); ++x)
                if (image.pixelColor(x, y) != background) { foundGrid = true; break; }
        QVERIFY(foundGrid);
    }

    void multiOutputTensorHoverUsesCopiedAnalysisText() {
        char error[512] = {};
        const QByteArray dataset = R"({"name":"Features","description":"","batch":{"inputs":{"features":{"dtype":"float32","shape":["B",4]}},"targets":{}}})";
        QVERIFY2(nn_app_create_dataset(application_, "test.features", "1.0.0", dataset.constData(),
                                       true, error, sizeof(error)), error);
        QVERIFY2(nn_app_set_parameter_text(application_, "source", "binding", "features",
                                           error, sizeof(error)), error);
        const QByteArray definition = R"({"name":"Dual typed source","description":"","kind":"layer","outputs":[{"id":"prediction","type":"output"},{"id":"objective","type":"loss"}],"parameters":{},"view":{"color":"#6688aa","width":190,"height":100}})";
        const QByteArray lua = "return function(context, parameters, services)\n"
            "  local input = context.inputs[1]\n"
            "  if not input then return { status = 'unresolved', message = 'missing' } end\n"
            "  return { status = 'success', outputs = { prediction = input, objective = input } }\n"
            "end";
        QVERIFY2(nn_app_create_stereotype(application_, "test.dual-hover", "1.0.0",
                                          definition.constData(), lua.constData(), "{}",
                                          error, sizeof(error)), error);
        QVERIFY2(nn_app_add_node(application_, "dual-hover", "test.dual-hover", "1.0.0", "",
                                 340, 100, error, sizeof(error)), error);
        QVERIFY2(nn_app_connect(application_, "dual-hover-input", "source", "out", "dual-hover", "in",
                                error, sizeof(error)), error);
        scene_->refresh();
        NodeItem *node = scene_->nodeItem(QStringLiteral("dual-hover"));
        QVERIFY(node);
        QTest::mouseMove(view_->viewport(), view_->mapFromScene(node->sceneBoundingRect().center()));
        pumpEvents();
        const QString copied = node->tensorSummary();
        QVERIFY(copied.contains(QStringLiteral("prediction")));
        QVERIFY(copied.contains(QStringLiteral("objective")));
        QVERIFY(copied.contains(QStringLiteral("float32")));
        QVERIFY(copied.contains(QStringLiteral("B, 4")));
        QGraphicsRectItem *popup = nullptr;
        for (QGraphicsItem *child : node->childItems())
            if (auto *rect = dynamic_cast<QGraphicsRectItem *>(child)) popup = rect;
        QVERIFY(popup && popup->isVisible());
        QVERIFY(popup->flags() & QGraphicsItem::ItemIgnoresTransformations);
        QTest::mouseMove(view_->viewport(), QPoint(8, 8));
        pumpEvents();
        QVERIFY(!popup->isVisible());
        QVERIFY2(nn_app_rename_node(application_, "dual-hover", "Renamed", error, sizeof(error)), error);
        QVERIFY(node->tensorSummary() == copied);
        scene_->refresh();
        node = scene_->nodeItem(QStringLiteral("dual-hover"));
        QVERIFY(node && node->tensorSummary() == copied);
    }

    void escapeCancelsConnectionDraft() {
        NodeItem *source = scene_->nodeItem(QStringLiteral("source"));
        QVERIFY(source);
        PortItem *out = nullptr;
        for (PortItem *port : source->ports())
            if (port->isOutput()) out = port;
        QVERIFY(out);
        const QPoint start = view_->mapFromScene(out->scenePos());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), start + QPoint(55, 50), 30);
        QTest::keyClick(view_, Qt::Key_Escape);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier,
                            start + QPoint(55, 50));
        pumpEvents();
        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(0));
    }

    void multiSelectionDragPersistsEveryNode() {
        NodeItem *source = scene_->nodeItem(QStringLiteral("source"));
        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(source && target);
        source->setSelected(true);
        target->setSelected(true);
        const QPointF sourceBefore = source->pos();
        const QPointF targetBefore = target->pos();
        const QPoint start = view_->mapFromScene(target->sceneBoundingRect().center());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), start + QPoint(31, 24), 30);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier,
                            start + QPoint(31, 24));
        pumpEvents();
        const NNModel *model = nn_app_model(application_);
        const NNNode *savedSource = nn_model_find_node(model, "source");
        const NNNode *savedTarget = nn_model_find_node(model, "target");
        QVERIFY(savedSource && savedTarget);
        QCOMPARE(savedSource->x, 140.0);
        QCOMPARE(savedSource->y, 120.0);
        QCOMPARE(savedTarget->x, 140.0);
        QCOMPARE(savedTarget->y, 300.0);
    }

    void escapeRestoresNodeDragPreview() {
        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(target);
        const QPointF before = target->pos();
        const QPoint start = view_->mapFromScene(target->sceneBoundingRect().center());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), start + QPoint(48, 35), 30);
        QTest::keyClick(view_, Qt::Key_Escape);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier,
                            start + QPoint(48, 35));
        pumpEvents();
        const NNNode *saved = nn_model_find_node(nn_app_model(application_), "target");
        QVERIFY(saved);
        QVERIFY(qAbs(saved->x - before.x()) < 1.0);
        QVERIFY(qAbs(saved->y - before.y()) < 1.0);
    }

    void cameraDoesNotChangeModelCoordinates() {
        const NNNode *before = nn_model_find_node(nn_app_model(application_), "target");
        QVERIFY(before);
        const QPointF modelPosition(before->x, before->y);
        const QPoint anchor(360, 260);
        const QPointF sceneAtAnchor = view_->mapToScene(anchor);
        QWheelEvent zoomIn(QPointF(anchor), QPointF(view_->viewport()->mapToGlobal(anchor)),
                           QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier,
                           Qt::NoScrollPhase, false);
        QApplication::sendEvent(view_->viewport(), &zoomIn);
        const QPointF afterZoom = view_->mapToScene(anchor);
        QVERIFY(QLineF(sceneAtAnchor, afterZoom).length() < 2.0);

        QTest::mousePress(view_->viewport(), Qt::MiddleButton, Qt::NoModifier, anchor);
        QTest::mouseMove(view_->viewport(), anchor + QPoint(72, 43), 30);
        QTest::mouseRelease(view_->viewport(), Qt::MiddleButton, Qt::NoModifier,
                            anchor + QPoint(72, 43));
        pumpEvents();
        const NNNode *after = nn_model_find_node(nn_app_model(application_), "target");
        QVERIFY(after);
        QCOMPARE(after->x, modelPosition.x());
        QCOMPARE(after->y, modelPosition.y());

        for (int i = 0; i < 40; ++i) {
            QWheelEvent event(QPointF(anchor), QPointF(view_->viewport()->mapToGlobal(anchor)),
                              QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier,
                              Qt::NoScrollPhase, false);
            QApplication::sendEvent(view_->viewport(), &event);
        }
        QVERIFY(view_->transform().m11() <= 3.001);
        for (int i = 0; i < 80; ++i) {
            QWheelEvent event(QPointF(anchor), QPointF(view_->viewport()->mapToGlobal(anchor)),
                              QPoint(), QPoint(0, -120), Qt::NoButton, Qt::NoModifier,
                              Qt::NoScrollPhase, false);
            QApplication::sendEvent(view_->viewport(), &event);
        }
        QVERIFY(view_->transform().m11() >= 0.199);
    }

    void positionedParameterRowsSizeCards() {
        NodeItem card(nullptr, QStringLiteral("positioned"), QStringLiteral("Positioned layer"),
                      QStringLiteral("custom.layer"), QPointF(), QColor("#48d1cc"),
                      {{QStringLiteral("stride"), QStringLiteral("2")}},
                      {{QStringLiteral("padding"), QStringLiteral("same")}});
        QCOMPARE(card.boundingRect().height(), 145.0);
        QVERIFY(card.boundingRect().width() >= 189.0);
        QImage image(card.boundingRect().size().toSize(), QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        QPainter painter(&image);
        card.paint(&painter, nullptr);
        painter.end();
        QVERIFY(!image.isNull());
        const QColor parameterBand = image.pixelColor(image.width() / 2, 20);
        const QColor titleBand = image.pixelColor(image.width() / 2, 72);
        QVERIFY(parameterBand.isValid());
        QVERIFY(titleBand.isValid());
        QVERIFY(parameterBand != titleBand);
        QVERIFY(image.pixelColor(5, 10).alpha() < parameterBand.alpha());
    }

    void typedBoundaryCirclesAndSeededTerminals() {
        scene_->refresh();
        const NNModel *model = nn_app_model(application_);
        bool hasRootOutput = false, hasRootLossOutput = false;
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node || (node->scope_id && *node->scope_id)) continue;
            hasRootOutput |= std::strcmp(node->package_id, "core.output") == 0;
            hasRootLossOutput |= std::strcmp(node->package_id, "core.loss-output") == 0;
        }
        QVERIFY(hasRootOutput);
        QVERIFY(hasRootLossOutput);
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node || (node->scope_id && *node->scope_id) || !node->package_id ||
                (std::strcmp(node->package_id, "core.output") != 0 &&
                 std::strcmp(node->package_id, "core.loss-output") != 0)) continue;
            NodeItem *terminal = scene_->nodeItem(QString::fromUtf8(node->id));
            QVERIFY(terminal);
            QVERIFY(!terminal->ports().isEmpty());
            for (PortItem *port : terminal->ports()) {
                const QImage portImage = renderPort(port);
                QVERIFY(port->shape().contains(QPointF(0, 0)));
                QCOMPARE(portImage.pixelColor(7, 7),
                         port->outputType() == QStringLiteral("loss") ? QColor("#c62828")
                            : port->isOutput() ? QColor("#161616") : QColor("#ffffff"));
                QCOMPARE(portImage.pixelColor(11, 5), QColor(Qt::white));
            }
        }
        NodeItem output(nullptr, QStringLiteral("output"), QStringLiteral("Prediction"),
                        QStringLiteral("core.output"), {}, QColor("#45a"));
        output.setBoundaryKind(QStringLiteral("output"));
        QVERIFY(output.boundingRect().contains(QPointF(47, 38)));
        QVERIFY(output.boundingRect().width() >= 160.0);
        QImage image(output.boundingRect().size().toSize(), QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        QPainter painter(&image);
        output.paint(&painter, nullptr);
        painter.end();
        QCOMPARE(image.pixelColor(47, 38), QColor("#8b5a2b"));
        output.setSelected(true);
        image.fill(Qt::transparent);
        QPainter selectedPainter(&image);
        output.paint(&selectedPainter, nullptr);
        selectedPainter.end();
        QCOMPARE(image.pixelColor(47, 38), QColor("#8b5a2b"));

        output.setProblemCategory(QStringLiteral("incomplete"));
        image.fill(Qt::transparent);
        QPainter diagnosticPainter(&image);
        output.paint(&diagnosticPainter, nullptr);
        diagnosticPainter.end();
        QCOMPARE(image.pixelColor(7, 11), QColor("#a66a12"));
        QCOMPARE(image.pixelColor(47, 38), QColor("#8b5a2b"));

        const QString longLabel = QStringLiteral("A boundary label long enough to need additional external space");
        NodeItem longOutput(nullptr, QStringLiteral("long-output"), longLabel,
                            QStringLiteral("core.output"), {}, QColor("#45a"));
        longOutput.setBoundaryKind(QStringLiteral("output"));
        QFont labelFont;
        labelFont.setBold(true);
        labelFont.setPointSizeF(11.0);
        const qreal labelWidth = QFontMetricsF(labelFont).horizontalAdvance(longLabel);
        QVERIFY(longOutput.boundingRect().width() >= 100.0 + labelWidth - 1.0);
        QImage longImage(longOutput.boundingRect().size().toSize(), QImage::Format_ARGB32_Premultiplied);
        longImage.fill(Qt::transparent);
        QPainter longPainter(&longImage);
        longOutput.paint(&longPainter, nullptr);
        longPainter.end();
        QVERIFY(!longImage.isNull());
    }

    void horizontalDirectionPlacesHandlesAndConnectsRightToLeft() {
        NodeItem *source = scene_->nodeItem(QStringLiteral("source"));
        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(source && target);
        PortItem *out = nullptr;
        PortItem *in = nullptr;
        for (PortItem *port : source->ports()) if (port->isOutput()) out = port;
        for (PortItem *port : target->ports()) if (!port->isOutput()) in = port;
        QVERIFY(out && in);
        QCOMPARE(out->y(), 60.0);
        QCOMPARE(in->y(), 9.0);
        char error[512] = {};
        QVERIFY2(nn_app_move_node(application_, "flow", 800, 400, error, sizeof(error)), error);
        QVERIFY2(nn_app_move_node(application_, "source", 400, 100, error, sizeof(error)), error);
        QVERIFY2(nn_app_move_node(application_, "target", 100, 100, error, sizeof(error)), error);
        const NNModel *model = nn_app_model(application_);
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node || !node->id || std::strcmp(node->id, "source") == 0 ||
                std::strcmp(node->id, "target") == 0) continue;
            const QByteArray nodeId(node->id);
            QVERIFY2(nn_app_move_node(application_, nodeId.constData(), 4000 + qreal(i) * 240,
                                      4000 + qreal(i) * 180, error, sizeof(error)), error);
        }
        scene_->refresh();
        view_->centerOn(QPointF(280, 150));
        source = scene_->nodeItem(QStringLiteral("source"));
        target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(source && target);
        out = nullptr;
        in = nullptr;
        for (PortItem *port : source->ports()) if (port->isOutput()) out = port;
        for (PortItem *port : target->ports()) if (!port->isOutput()) in = port;
        QVERIFY(out && in);

        scene_->setFlowDirection(FlowDirection::Horizontal);
        QCOMPARE(out->x(), 25.0);
        QCOMPARE(in->x(), target->boundingRect().width() - 8.0);
        QVERIFY(out->scenePos().x() > in->scenePos().x());
        const QPoint start = view_->mapFromScene(out->scenePos());
        const QPoint end = view_->mapFromScene(in->scenePos());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), end, 30);
        QGraphicsPathItem *draft = nullptr;
        for (QGraphicsItem *item : scene_->items()) {
            if (dynamic_cast<EdgeItem *>(item)) continue;
            if (auto *path = dynamic_cast<QGraphicsPathItem *>(item)) draft = path;
        }
        QVERIFY(draft);
        const QPainterPath draftPath = draft->path();
        QVERIFY(draftPath.elementCount() >= 2);
        for (int i = 1; i < draftPath.elementCount(); ++i) {
            const auto a = draftPath.elementAt(i - 1), b = draftPath.elementAt(i);
            QVERIFY(qFuzzyCompare(a.x + 1.0, b.x + 1.0) || qFuzzyCompare(a.y + 1.0, b.y + 1.0));
        }
        QVERIFY(draftPath.elementAt(1).x < draftPath.elementAt(0).x);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier, end);
        pumpEvents();
        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(1));
        const NNEdge *connected = nn_model_edge_at(nn_app_model(application_), 0);
        QVERIFY(connected);
        EdgeItem *edge = scene_->edgeItem(QString::fromUtf8(connected->id));
        QVERIFY(edge);
        source = scene_->nodeItem(QStringLiteral("source"));
        target = scene_->nodeItem(QStringLiteral("target"));
        out = nullptr;
        in = nullptr;
        for (PortItem *port : source->ports()) if (port->isOutput()) out = port;
        for (PortItem *port : target->ports()) if (!port->isOutput()) in = port;
        const QVector<QPointF> horizontalRoute = edge->routePoints();
        QVERIFY(horizontalRoute.size() >= 2);
        for (int i = 1; i < horizontalRoute.size(); ++i)
            QVERIFY(qFuzzyCompare(horizontalRoute[i - 1].x() + 1.0, horizontalRoute[i].x() + 1.0) ||
                    qFuzzyCompare(horizontalRoute[i - 1].y() + 1.0, horizontalRoute[i].y() + 1.0));
        QVERIFY(horizontalRoute[1].x() < horizontalRoute[0].x());

        scene_->setFlowDirection(FlowDirection::Vertical);
        QCOMPARE(out->y(), 60.0);
        QCOMPARE(in->y(), 9.0);
        const QVector<QPointF> verticalRoute = edge->routePoints();
        QVERIFY(verticalRoute.size() >= 2);
        for (int i = 1; i < verticalRoute.size(); ++i)
            QVERIFY(qFuzzyCompare(verticalRoute[i - 1].x() + 1.0, verticalRoute[i].x() + 1.0) ||
                    qFuzzyCompare(verticalRoute[i - 1].y() + 1.0, verticalRoute[i].y() + 1.0));
        QVERIFY(verticalRoute[1].y() > verticalRoute[0].y());
    }

    void genericJoinJunctionResizesWithoutDroppingConnections() {
        const QByteArray definition = R"({"name":"Custom merge","description":"","kind":"join","outputs":[{"id":"out","type":"output"}],"parameters":{},"view":{"color":"#6688aa","width":190,"height":100}})";
        const QByteArray lua = "return function(context, parameters, services) return { status = 'unresolved', message = 'No inputs' } end";
        char error[512] = {};
        QVERIFY2(nn_app_create_stereotype(application_, "test.custom-join", "1.0.0",
                                          definition.constData(), lua.constData(), "{}",
                                          error, sizeof(error)), error);
        QVERIFY2(nn_app_add_node(application_, "custom-join", "test.custom-join", "1.0.0", "",
                                 420, 220, error, sizeof(error)), error);
        QVERIFY2(nn_app_connect(application_, "join-input-one", "source", "out", "custom-join",
                                "in-1", error, sizeof(error)), error);
        QVERIFY2(nn_app_connect(application_, "join-input-two", "source", "out", "custom-join",
                                "in-2", error, sizeof(error)), error);
        scene_->refresh();
        NodeItem *join = scene_->nodeItem(QStringLiteral("custom-join"));
        QVERIFY(join);
        QCOMPARE(join->ports().size(), 3);
        QVERIFY(join->boundingRect().contains(join->joinAddControlRect()));
        QVERIFY(join->boundingRect().contains(join->joinRemoveControlRect()));
        QVERIFY(join->joinRemoveControlRect().center().x() < join->joinAddControlRect().center().x());
        QStringList visibleInputs;
        for (PortItem *port : join->ports())
            if (!port->isOutput()) visibleInputs.append(port->handleId());
        QCOMPARE(visibleInputs, QStringList({QStringLiteral("in-1"), QStringLiteral("in-2")}));
        QImage junctionImage(join->boundingRect().size().toSize(), QImage::Format_ARGB32_Premultiplied);
        junctionImage.fill(Qt::transparent);
        QPainter junctionPainter(&junctionImage);
        junctionPainter.translate(-join->boundingRect().topLeft());
        join->paint(&junctionPainter, nullptr);
        junctionPainter.end();
        QCOMPARE(junctionImage.pixelColor(5, 5).alpha(), 0);
        QCOMPARE(junctionImage.pixelColor(int(join->junctionRect().center().x()),
                                         int(join->junctionRect().center().y())), QColor(32, 39, 48));

        const QPoint add = view_->mapFromScene(join->mapToScene(join->joinAddControlRect().center()));
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, add);
        pumpEvents();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QVERIFY(join);
        PortItem *joinInputThree = nullptr;
        for (PortItem *port : join->ports())
            if (!port->isOutput() && port->handleId() == QStringLiteral("in-3")) joinInputThree = port;
        QVERIFY(joinInputThree);
        QCOMPARE(join->ports().size(), 4);
        scene_->refresh();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QVERIFY(std::any_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-3");
        }));

        PortItem *sourceOutput = nullptr;
        for (PortItem *port : scene_->nodeItem(QStringLiteral("source"))->ports())
            if (port->isOutput()) sourceOutput = port;
        QVERIFY(sourceOutput);
        joinInputThree = nullptr;
        for (PortItem *port : scene_->nodeItem(QStringLiteral("custom-join"))->ports())
            if (!port->isOutput() && port->handleId() == QStringLiteral("in-3")) joinInputThree = port;
        QVERIFY(joinInputThree);
        const QPoint start = view_->mapFromScene(sourceOutput->scenePos());
        const QPoint end = view_->mapFromScene(joinInputThree->scenePos());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), end, 30);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier, end);
        pumpEvents();
        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(3));
        bool connectedThree = false;
        for (size_t i = 0; i < nn_model_edge_count(nn_app_model(application_)); ++i) {
            const NNEdge *edge = nn_model_edge_at(nn_app_model(application_), i);
            connectedThree |= edge && QString::fromUtf8(edge->target_handle_id) == QStringLiteral("in-3");
        }
        QVERIFY(connectedThree);
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QCOMPARE(join->ports().size(), 4);
        QVERIFY(std::none_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-4");
        }));

        const QPoint addFourth = view_->mapFromScene(join->mapToScene(join->joinAddControlRect().center()));
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, addFourth);
        pumpEvents();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QVERIFY(std::any_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-4");
        }));

        scene_->setFlowDirection(FlowDirection::Horizontal);
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        PortItem *horizontalInput = nullptr, *horizontalOutput = nullptr;
        for (PortItem *port : join->ports()) {
            if (port->isOutput()) horizontalOutput = port;
            else if (port->handleId() == QStringLiteral("in-1")) horizontalInput = port;
        }
        QVERIFY(horizontalInput && horizontalOutput);
        QVERIFY(horizontalInput->x() > join->boundingRect().center().x());
        QVERIFY(horizontalOutput->x() < join->boundingRect().center().x());
        QVERIFY(join->joinRemoveControlRect().center().y() < join->joinAddControlRect().center().y());

        const QPoint remove = view_->mapFromScene(join->mapToScene(join->joinRemoveControlRect().center()));
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, remove);
        pumpEvents();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        bool retainedOne = false, retainedThree = false, exposedFour = false;
        for (PortItem *port : join->ports()) {
            if (port->isOutput()) continue;
            retainedOne |= port->handleId() == QStringLiteral("in-1");
            retainedThree |= port->handleId() == QStringLiteral("in-3");
            exposedFour |= port->handleId() == QStringLiteral("in-4");
        }
        QVERIFY(retainedOne && retainedThree);
        QVERIFY(!exposedFour);
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier,
            view_->mapFromScene(join->mapToScene(join->joinRemoveControlRect().center())));
        pumpEvents();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QVERIFY(std::any_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-3");
        }));

        QByteArray thirdEdgeId;
        for (size_t i = 0; i < nn_model_edge_count(nn_app_model(application_)); ++i) {
            const NNEdge *edge = nn_model_edge_at(nn_app_model(application_), i);
            if (edge && QString::fromUtf8(edge->target_handle_id) == QStringLiteral("in-3"))
                thirdEdgeId = edge->id;
        }
        QVERIFY(!thirdEdgeId.isEmpty());
        QVERIFY2(nn_app_disconnect(application_, thirdEdgeId.constData(), error, sizeof(error)), error);
        scene_->refresh();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier,
            view_->mapFromScene(join->mapToScene(join->joinRemoveControlRect().center())));
        pumpEvents();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QCOMPARE(join->ports().size(), 3);
        QVERIFY(std::none_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-3");
        }));
        const QPoint addAgain = view_->mapFromScene(join->mapToScene(join->joinAddControlRect().center()));
        QTest::mouseClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, addAgain);
        pumpEvents();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QVERIFY(std::any_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-3");
        }));
        QVERIFY2(nn_app_save(application_, error, sizeof(error)), error);
        QVERIFY2(nn_app_open(application_, projectPath_.constData(), error, sizeof(error)), error);
        scene_->refresh();
        join = scene_->nodeItem(QStringLiteral("custom-join"));
        QCOMPARE(join->ports().size(), 3);
        QVERIFY(std::none_of(join->ports().cbegin(), join->ports().cend(), [](PortItem *port) {
            return !port->isOutput() && port->handleId() == QStringLiteral("in-3");
        }));
        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(2));
    }

    void terminalRimPortsFollowBothDirections() {
        NodeItem *input = scene_->nodeItem(QStringLiteral("source"));
        QVERIFY(input);
        PortItem *inputOutput = nullptr;
        for (PortItem *port : input->ports()) if (port->isOutput()) inputOutput = port;
        QVERIFY(inputOutput);
        QCOMPARE(inputOutput->y(), 60.0);

        NodeItem *output = nullptr;
        const NNModel *model = nn_app_model(application_);
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (node && std::strcmp(node->package_id, "core.output") == 0 &&
                (!node->scope_id || !*node->scope_id))
                output = scene_->nodeItem(QString::fromUtf8(node->id));
        }
        QVERIFY(output);
        const QByteArray outputId = output->id().toUtf8();
        PortItem *terminalInput = nullptr;
        for (PortItem *port : output->ports()) if (!port->isOutput()) terminalInput = port;
        QVERIFY(terminalInput);
        QCOMPARE(terminalInput->y(), 16.0);
        char error[512] = {};
        QVERIFY2(nn_app_move_node(application_, "flow", 800, 400, error, sizeof(error)), error);
        QVERIFY2(nn_app_move_node(application_, "source", 400, 100, error, sizeof(error)), error);
        QVERIFY2(nn_app_move_node(application_, outputId.constData(), 100, 100, error, sizeof(error)), error);
        scene_->refresh();
        view_->centerOn(QPointF(280, 150));
        input = scene_->nodeItem(QStringLiteral("source"));
        output = scene_->nodeItem(QString::fromUtf8(outputId));
        QVERIFY(input && output);
        inputOutput = nullptr;
        terminalInput = nullptr;
        for (PortItem *port : input->ports()) if (port->isOutput()) inputOutput = port;
        for (PortItem *port : output->ports()) if (!port->isOutput()) terminalInput = port;
        QVERIFY(inputOutput && terminalInput);

        scene_->setFlowDirection(FlowDirection::Horizontal);
        QCOMPARE(inputOutput->x(), 25.0);
        QCOMPARE(terminalInput->x(), 69.0);
        QVERIFY(inputOutput->scenePos().x() > terminalInput->scenePos().x());
        const QPoint start = view_->mapFromScene(inputOutput->scenePos());
        const QPoint end = view_->mapFromScene(terminalInput->scenePos());
        QTest::mousePress(view_->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view_->viewport(), end, 30);
        QTest::mouseRelease(view_->viewport(), Qt::LeftButton, Qt::NoModifier, end);
        pumpEvents();
        QCOMPARE(nn_model_edge_count(nn_app_model(application_)), size_t(1));
        const NNEdge *edge = nn_model_edge_at(nn_app_model(application_), 0);
        QVERIFY(edge);
        QCOMPARE(QString::fromUtf8(edge->target_id), QString::fromUtf8(outputId));

        scene_->setFlowDirection(FlowDirection::Vertical);
        input = scene_->nodeItem(QStringLiteral("source"));
        output = scene_->nodeItem(QString::fromUtf8(outputId));
        inputOutput = nullptr;
        terminalInput = nullptr;
        for (PortItem *port : input->ports()) if (port->isOutput()) inputOutput = port;
        for (PortItem *port : output->ports()) if (!port->isOutput()) terminalInput = port;
        QCOMPARE(inputOutput->y(), 60.0);
        QCOMPARE(terminalInput->y(), 16.0);
    }

    void fitFramesLargeBoundsAndRecentersAfterPan() {
        QVERIFY(addNode("far", "core.relu", "", 200000, 200000));
        scene_->refresh();
        view_->horizontalScrollBar()->setValue(view_->horizontalScrollBar()->maximum());
        view_->verticalScrollBar()->setValue(view_->verticalScrollBar()->maximum());
        view_->fitGraph();
        const QRectF bounds = scene_->itemsBoundingRect();
        QVERIFY(view_->transform().m11() < 0.2);
        const QPoint mappedCenter = view_->mapFromScene(bounds.center());
        const QPoint viewportCenter = view_->viewport()->rect().center();
        const qreal centerError = QLineF(mappedCenter, viewportCenter).length();
        QVERIFY2(centerError < 2.0,
                 qPrintable(QStringLiteral("fit center error %1 px (mapped %2,%3; viewport %4,%5; zoom %6)")
                    .arg(centerError).arg(mappedCenter.x()).arg(mappedCenter.y())
                    .arg(viewportCenter.x()).arg(viewportCenter.y()).arg(view_->transform().m11())));
    }

    void inputWithTwoTypedOutputsHasDistinctConnectableHotspots() {
        const QByteArray definition = R"({"name":"Dual input","description":"","kind":"input","outputs":[{"id":"prediction","type":"output"},{"id":"objective","type":"loss"}],"parameters":{},"view":{"color":"#6688aa","width":190,"height":100}})";
        const QByteArray lua = "return function(context, parameters, services) return { status = 'unresolved', message = 'No dataset binding' } end";
        char error[512] = {};
        QVERIFY2(nn_app_create_stereotype(application_, "test.dual-input", "1.0.0",
                                          definition.constData(), lua.constData(), "{}",
                                          error, sizeof(error)), error);
        QVERIFY2(nn_app_add_node(application_, "dual-input", "test.dual-input", "1.0.0", "",
                                 360, 180, error, sizeof(error)), error);
        QString outputId, lossOutputId;
        const NNModel *model = nn_app_model(application_);
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node || (node->scope_id && *node->scope_id)) continue;
            if (std::strcmp(node->package_id, "core.output") == 0) outputId = QString::fromUtf8(node->id);
            if (std::strcmp(node->package_id, "core.loss-output") == 0) lossOutputId = QString::fromUtf8(node->id);
        }
        QVERIFY(!outputId.isEmpty() && !lossOutputId.isEmpty());
        scene_->refresh();
        NodeItem *input = scene_->nodeItem(QStringLiteral("dual-input"));
        QVERIFY(input);
        QCOMPARE(input->boundaryKind(), QStringLiteral("input"));
        PortItem *outputPort = nullptr, *lossPort = nullptr;
        for (PortItem *port : input->ports()) {
            if (!port->isOutput()) continue;
            if (port->handleId() == QStringLiteral("prediction")) outputPort = port;
            if (port->handleId() == QStringLiteral("objective")) lossPort = port;
        }
        QVERIFY(outputPort && lossPort);
        for (PortItem *port : {outputPort, lossPort}) {
            const QImage portImage = renderPort(port);
            QCOMPARE(portImage.pixelColor(7, 7),
                     port == outputPort ? QColor("#161616") : QColor("#c62828"));
            QCOMPARE(portImage.pixelColor(11, 5), QColor(Qt::white));
        }
        const QImage atRest = renderPort(outputPort);
        QTest::mouseMove(view_->viewport(), view_->mapFromScene(outputPort->scenePos()));
        pumpEvents();
        QVERIFY(renderPort(outputPort) != atRest);
        QVERIFY(QLineF(outputPort->pos(), lossPort->pos()).length() > 14.0);
        scene_->setFlowDirection(FlowDirection::Horizontal);
        QVERIFY(outputPort->x() < input->boundingRect().center().x());
        QVERIFY(lossPort->x() < input->boundingRect().center().x());
        QVERIFY(qAbs(outputPort->y() - lossPort->y()) > 10.0);
        scene_->setFlowDirection(FlowDirection::Vertical);
        QVERIFY(outputPort->toolTip().contains(QStringLiteral("prediction")));
        QVERIFY(lossPort->toolTip().contains(QStringLiteral("loss")));
        QVERIFY2(nn_app_connect(application_, "dual-output-edge", "dual-input", "prediction",
                                outputId.toUtf8().constData(), "in", error, sizeof(error)), error);
        QVERIFY2(nn_app_connect(application_, "dual-loss-edge", "dual-input", "objective",
                                lossOutputId.toUtf8().constData(), "in", error, sizeof(error)), error);
        scene_->refresh();
        QCOMPARE(scene_->edgeItem(QStringLiteral("dual-output-edge"))->pen().color(), QColor("#111111"));
        QCOMPARE(scene_->edgeItem(QStringLiteral("dual-loss-edge"))->pen().color(), QColor("#c62828"));

        NodeItem inputLabel(nullptr, QStringLiteral("label-test"), QStringLiteral("image"),
                            QStringLiteral("test.dual-input"), {}, QColor("#6688aa"));
        inputLabel.setBoundaryKind(QStringLiteral("input"));
        QImage labelImage(inputLabel.boundingRect().size().toSize(), QImage::Format_ARGB32_Premultiplied);
        labelImage.fill(Qt::transparent);
        QPainter labelPainter(&labelImage);
        inputLabel.paint(&labelPainter, nullptr);
        labelPainter.end();
        int lastLabelInkRow = -1;
        for (int y = 0; y < labelImage.height(); ++y)
            for (int x = 82; x < labelImage.width(); ++x)
                if (labelImage.pixelColor(x, y).alpha() > 0) lastLabelInkRow = qMax(lastLabelInkRow, y);
        QVERIFY(lastLabelInkRow >= 0);
        // The upper typed hotspot for two outputs is at y=26; external text
        // must stay above it, not on the straight outgoing edge band.
        QVERIFY(lastLabelInkRow < 24);
    }

    void longOutputHandleIsEnumeratedAndConnectable() {
        const QString longHandle = QStringLiteral("feature_") + QString(180, QLatin1Char('x'));
        const QJsonObject output{{QStringLiteral("id"), longHandle},
                                 {QStringLiteral("type"), QStringLiteral("output")}};
        const QJsonObject definition{
            {QStringLiteral("name"), QStringLiteral("Long handle source")},
            {QStringLiteral("description"), QString()},
            {QStringLiteral("kind"), QStringLiteral("layer")},
            {QStringLiteral("outputs"), QJsonArray{output}},
            {QStringLiteral("parameters"), QJsonObject{}},
            {QStringLiteral("view"), QJsonObject{{QStringLiteral("color"), QStringLiteral("#6688aa")},
                                                  {QStringLiteral("width"), 190},
                                                  {QStringLiteral("height"), 100}}}};
        const QByteArray json = QJsonDocument(definition).toJson(QJsonDocument::Compact);
        const QByteArray lua = "return function(context, parameters, services) return { status = 'unresolved', message = 'No input' } end";
        char error[512] = {};
        QVERIFY2(nn_app_create_stereotype(application_, "test.long-output", "1.0.0",
                                          json.constData(), lua.constData(), "{}",
                                          error, sizeof(error)), error);
        QVERIFY2(nn_app_add_node(application_, "long-source", "test.long-output", "1.0.0", "",
                                 360, 180, error, sizeof(error)), error);
        QVERIFY2(nn_app_add_node(application_, "long-target", "core.relu", "0.1.0", "",
                                 620, 180, error, sizeof(error)), error);
        QVERIFY2(nn_app_connect(application_, "long-output-edge", "long-source",
                                longHandle.toUtf8().constData(), "long-target", "in",
                                error, sizeof(error)), error);
        scene_->refresh();
        NodeItem *source = scene_->nodeItem(QStringLiteral("long-source"));
        QVERIFY(source);
        PortItem *longPort = nullptr;
        for (PortItem *port : source->ports())
            if (port->handleId() == longHandle) longPort = port;
        QVERIFY(longPort);
        QCOMPARE(longPort->handleId(), longHandle);
        QVERIFY(scene_->edgeItem(QStringLiteral("long-output-edge")));
        scene_->refresh();
        source = scene_->nodeItem(QStringLiteral("long-source"));
        QVERIFY(source);
        longPort = nullptr;
        for (PortItem *port : source->ports())
            if (port->handleId() == longHandle) longPort = port;
        QVERIFY(longPort);
        QVERIFY(scene_->edgeItem(QStringLiteral("long-output-edge")));
    }
};

QTEST_MAIN(QtCanvasTest)
#include "qt_canvas_test.moc"
