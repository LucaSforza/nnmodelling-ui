#include "EdgeItem.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "SubflowItem.hpp"

#include "application/application.h"
#include "model/model.h"

#include <QTemporaryDir>
#include <QImage>
#include <QPainter>
#include <QFontMetricsF>
#include <QLineF>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QUuid>
#include <QWheelEvent>
#include <QtTest>

#include <memory>
#include <cstring>

class QtCanvasTest final : public QObject {
    Q_OBJECT
private:
    std::unique_ptr<QTemporaryDir> directory_;
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
        QVERIFY(qAbs(saved->x - 138.0) < 1.0);
        QVERIFY(qAbs(saved->y - 307.0) < 1.0);
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
        QSignalSpy scopeSpy(scene_, &GraphScene::scopeChanged);
        const QPoint center = view_->mapFromScene(flow->sceneBoundingRect().center());
        QTest::mouseDClick(view_->viewport(), Qt::LeftButton, Qt::NoModifier, center);
        pumpEvents();
        QCOMPARE(scene_->scope(), QStringLiteral("flow"));
        QCOMPARE(scopeSpy.count(), 1);
        QVERIFY(scene_->nodeItem(QStringLiteral("child")));
        scene_->goToParentScope();
        pumpEvents();
        QVERIFY(scene_->scope().isEmpty());

        NodeItem *target = scene_->nodeItem(QStringLiteral("target"));
        QVERIFY(target);
        target->setSelected(true);
        QCOMPARE(scene_->selectedNodeId(), QStringLiteral("target"));
        scene_->deleteSelection();
        pumpEvents();
        QVERIFY(nn_model_find_node(nn_app_model(application_), "target") == nullptr);
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
        QVERIFY(qAbs(savedSource->x - (sourceBefore.x() + 31.0)) < 1.0);
        QVERIFY(qAbs(savedSource->y - (sourceBefore.y() + 24.0)) < 1.0);
        QVERIFY(qAbs(savedTarget->x - (targetBefore.x() + 31.0)) < 1.0);
        QVERIFY(qAbs(savedTarget->y - (targetBefore.y() + 24.0)) < 1.0);
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
            for (PortItem *port : terminal->ports()) QVERIFY(port->glyphSuppressed());
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
        QVERIFY(outputPort->glyphSuppressed() && lossPort->glyphSuppressed());
        QVERIFY(QLineF(outputPort->pos(), lossPort->pos()).length() > 14.0);
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
