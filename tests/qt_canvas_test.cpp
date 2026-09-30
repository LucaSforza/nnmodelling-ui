#include "EdgeItem.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "SubflowItem.hpp"

#include "application.h"
#include "model.h"

#include <QTemporaryDir>
#include <QImage>
#include <QPainter>
#include <QUuid>
#include <QWheelEvent>
#include <QtTest>

#include <memory>

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
    }
};

QTEST_MAIN(QtCanvasTest)
#include "qt_canvas_test.moc"
