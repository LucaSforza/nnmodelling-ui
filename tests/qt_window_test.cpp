#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "application.h"

#include <QAction>
#include <QApplication>
#include <QLineEdit>
#include <QMessageBox>
#include <QAbstractButton>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QTreeWidget>
#include <cstring>

class WindowTest : public QObject {
    Q_OBJECT
private slots:
    void lifecycleAndInspector();
};

static void answerDialog(QMessageBox::StandardButton answer) {
    QTimer::singleShot(0, [answer] {
        auto *box = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        if (box && box->button(answer)) box->button(answer)->click();
    });
}

void WindowTest::lifecycleAndInspector() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "qt-test", "Qt test", true,
                          error, sizeof(error)), error);
    MainWindow window(app); // Window owns the C application.
    window.show();
    QTest::qWait(30);
    auto *scene = window.findChild<GraphScene *>();
    auto *view = window.findChild<GraphView *>();
    auto *palette = window.findChild<QTreeWidget *>("palette");
    auto *inspector = window.findChild<QTreeWidget *>("inspector");
    auto *diagnostics = window.findChild<QTreeWidget *>("diagnostics");
    QVERIFY(scene && view && palette && inspector && diagnostics);
    QVERIFY(palette->topLevelItemCount() > 0);
    QCOMPARE(diagnostics->topLevelItemCount(), 8);
    QVERIFY(scene->nodeItem("dense1"));

    // Failed project replacement must keep the active graph and scene.
    answerDialog(QMessageBox::Ok);
    QVERIFY(!window.openProject(temporary.path() + "/missing"));
    QCOMPARE(nn_model_node_count(nn_app_model(app)), size_t(8));
    QVERIFY(scene->nodeItem("dense1"));

    scene->nodeItem("dense1")->setSelected(true);
    QCoreApplication::processEvents();
    QVERIFY(inspector->topLevelItemCount() >= 3);
    auto *name = qobject_cast<QLineEdit *>(inspector->itemWidget(inspector->topLevelItem(0), 1));
    QVERIFY(name);
    name->setFocus();
    name->selectAll();
    QTest::keyClicks(name, "Hidden layer");
    QTest::keyClick(name, Qt::Key_Return);
    QTRY_COMPARE(QString::fromUtf8(nn_model_find_node(nn_app_model(app), "dense1")->label),
                 QString("Hidden layer"));
    QCoreApplication::processEvents();
    QVERIFY(nn_project_dirty(nn_app_project(app)));

    // Exact integer text survives beyond the range of Qt's 32-bit spin box.
    auto *parameters = inspector->topLevelItem(2);
    QLineEdit *outFeatures = nullptr;
    for (int i = 0; i < parameters->childCount(); ++i) {
        auto *row = parameters->child(i);
        if (row->text(0) == "out_features")
            outFeatures = qobject_cast<QLineEdit *>(inspector->itemWidget(row, 1));
    }
    QVERIFY(outFeatures);
    outFeatures->setFocus();
    outFeatures->selectAll();
    QTest::keyClicks(outFeatures, "2147483648");
    QTest::keyClick(outFeatures, Qt::Key_Return);
    QCoreApplication::processEvents();
    char *text = nn_app_parameter_text(app, "dense1", "out_features");
    QVERIFY(text);
    QCOMPARE(QByteArray(text), QByteArray("2147483648"));
    nn_app_free_text(text);
    // Restore the valid fixture for the rendered evidence.
    QVERIFY(nn_app_set_parameter_text(app, "dense1", "out_features", "128", error, sizeof(error)));
    for (QAction *action : window.findChildren<QAction *>())
        if (action->text() == "Arrange") action->trigger();
    QCoreApplication::processEvents();
    view->fitGraph();
    const QString capture = qEnvironmentVariable("NN_TEST_CAPTURE");
    if (!capture.isEmpty()) QVERIFY(window.grab().save(capture));

    // Closing the window must respect cancellation and save before exit.
    answerDialog(QMessageBox::Cancel);
    QVERIFY(!window.close());
    QVERIFY(window.isVisible());
    answerDialog(QMessageBox::Save);
    QVERIFY(window.close());
    const QByteArray savedPath = (temporary.path() + "/qt-test").toUtf8();
    NNProject *saved = nn_project_open(savedPath.constData(),
        NN_SOURCE_DIR "/stereotype-packages/core", error, sizeof(error));
    QVERIFY2(saved, error);
    const NNNode *node = nn_model_find_node(nn_project_model(saved), "dense1");
    QVERIFY(node && std::strcmp(node->label, "Hidden layer") == 0);
    nn_project_close(saved);
}

QTEST_MAIN(WindowTest)
#include "qt_window_test.moc"
