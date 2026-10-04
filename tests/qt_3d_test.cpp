#include "Network3DView.hpp"

#include "application/application.h"
#include "project/project.h"

#include <QPushButton>
#include <QLabel>
#include <QToolButton>
#include <QTemporaryDir>
#include <QtTest>

class Network3DTest final : public QObject {
    Q_OBJECT
private slots:
    void rebuildFitHomeAndPickAreReadOnly();
};

void Network3DTest::rebuildFitHomeAndPickAreReadOnly() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *application = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(application);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(application, parent.constData(), "3d-test", "3D test", false,
                           error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(application, "dense-1", "core.relu", "0.1.0", "", 100, 100,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(application, "dense-2", "core.relu", "0.1.0", "", 360, 240,
                             error, sizeof(error)), error);
    const NNProject *project = nn_app_project(application);
    QVERIFY(project);
    nn_project_set_dirty(const_cast<NNProject *>(project), false);

    Network3DView view;
    view.resize(800, 600);
    view.show();
    QCoreApplication::processEvents();
    QSignalSpy ready(&view, &Network3DView::sceneReady);
    QVERIFY(view.rebuild(project));
    QVERIFY(view.selectedPath().isEmpty());
    QVERIFY(ready.count() == 1 && ready.takeFirst().at(0).toBool());
    QVERIFY(!nn_project_dirty(project));

    auto *fit = view.findChild<QToolButton *>("network3DFit");
    auto *home = view.findChild<QToolButton *>("network3DHome");
    QVERIFY(fit && home);
    QTest::mouseClick(fit, Qt::LeftButton);
    QTest::mouseClick(home, Qt::LeftButton);
    const QImage beforeFlight = view.grab().toImage();
    QTest::keyPress(&view, Qt::Key_W);
    QTest::qWait(35);
    QTest::keyRelease(&view, Qt::Key_W);
    QVERIFY(view.grab().toImage() != beforeFlight);
    const QImage beforeLook = view.grab().toImage();
    const QPoint lookStart = view.sceneViewportRect().center();
    QTest::mousePress(&view, Qt::RightButton, Qt::NoModifier, lookStart);
    QTest::mouseMove(&view, lookStart + QPoint(35, 12));
    QTest::mouseRelease(&view, Qt::RightButton, Qt::NoModifier, lookStart + QPoint(35, 12));
    QVERIFY(view.grab().toImage() != beforeLook);
    QTest::mouseClick(home, Qt::LeftButton);
    const QImage beforeTurn = view.grab().toImage();
    QTest::keyClick(&view, Qt::Key_Left);
    QVERIFY(view.grab().toImage() != beforeTurn);
    QTest::mouseClick(home, Qt::LeftButton);

    NN3DScene *scene = nn_3d_build(project, error, sizeof(error));
    QVERIFY2(scene, error);
    NN3DCamera camera{};
    const QRect canvas = view.sceneViewportRect();
    QVERIFY(canvas.width() > 0 && canvas.height() > 0);
    view.fitWholeGraph();
    nn_3d_camera_fit(scene, &camera, canvas.width() / double(canvas.height()));
    NN3DFrame frame{};
    QVERIFY(nn_3d_frame(scene, &camera, canvas.width(), canvas.height(), &frame));
    size_t secondIndex = SIZE_MAX;
    for (size_t i = 0; i < nn_3d_node_count(scene); ++i) {
        const NN3DNode *node = nn_3d_node_at(scene, i);
        if (node && node->source_id && QString::fromUtf8(node->source_id) == QStringLiteral("dense-2"))
            secondIndex = i;
    }
    QVERIFY(secondIndex != SIZE_MAX);
    QPoint pickPoint(-1, -1);
    size_t picked = SIZE_MAX;
    for (size_t i = 0; i < frame.count && picked != secondIndex; ++i) {
        const NN3DPrimitive &primitive = frame.items[i];
        if (primitive.kind != NN_3D_FACE || primitive.node != secondIndex) continue;
        const QPoint candidate(qRound((primitive.x[0] + primitive.x[1] + primitive.x[2] + primitive.x[3]) / 4.0),
                               qRound((primitive.y[0] + primitive.y[1] + primitive.y[2] + primitive.y[3]) / 4.0));
        if (!QRect(0, 0, canvas.width(), canvas.height()).contains(candidate)) continue;
        picked = nn_3d_pick(&frame, candidate.x(), candidate.y());
        if (picked == secondIndex) pickPoint = candidate;
    }
    QVERIFY(pickPoint.x() >= 0);
    QTest::mouseClick(&view, Qt::LeftButton, Qt::NoModifier, pickPoint + canvas.topLeft());
    const QString expectedPath = QString::fromUtf8(nn_3d_node_at(scene, secondIndex)->path);
    QVERIFY2(view.selectedPath() == expectedPath,
             qPrintable(QStringLiteral("picked '%1', expected '%2' at (%3,%4) in %5")
                            .arg(view.selectedPath(), expectedPath)
                            .arg(pickPoint.x()).arg(pickPoint.y())
                            .arg(QStringLiteral("%1x%2+%3+%4").arg(canvas.width()).arg(canvas.height())
                                     .arg(canvas.x()).arg(canvas.y()))));
    nn_3d_frame_dispose(&frame);
    nn_3d_free(scene);
    QVERIFY(!nn_project_dirty(project));

    QVERIFY(view.rebuild(nullptr) == false);
    QCoreApplication::processEvents();
    QVERIFY(view.findChild<QLabel *>("network3DError")->isVisible());
    QVERIFY(view.selectedPath().isEmpty());
    QVERIFY(ready.count() == 1 && !ready.takeFirst().at(0).toBool());
    nn_app_free(application);
}

QTEST_MAIN(Network3DTest)
#include "qt_3d_test.moc"
