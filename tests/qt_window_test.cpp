#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "application.h"
#include "catalog.h"
#include "inference.h"
#include "project.h"

#include <QAction>
#include <QApplication>
#include <QLineEdit>
#include <QComboBox>
#include <QPushButton>
#include <QMessageBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QEventLoop>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QTableWidget>
#include <QPlainTextEdit>
#include <QPalette>
#include <cmath>
#include <QAbstractButton>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QTreeWidget>
#include <cstring>
#include <functional>

class WindowTest : public QObject {
    Q_OBJECT
private slots:
    void lifecycleAndInspector();
    void visualResourceAuthoringForms();
    void automationRejectsInvalidScopeAndCapturesCurrentScope();
    void socketOptionRequiresPath();
};

static void answerDialog(QMessageBox::StandardButton answer) {
    QTimer::singleShot(0, [answer] {
        auto *box = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        if (box && box->button(answer)) box->button(answer)->click();
    });
}

static qreal contrastAgainstWhite(const QColor &color) {
    auto channel = [](int value) {
        const qreal c = value / 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    };
    const qreal luminance = 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) +
                            0.0722 * channel(color.blue());
    return 1.05 / (luminance + 0.05);
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
    const QPalette systemPalette = QApplication::palette();
    QPalette darkDesktop = systemPalette;
    darkDesktop.setColor(QPalette::Window, QColor("#20242b"));
    darkDesktop.setColor(QPalette::WindowText, QColor("#f0f2f5"));
    darkDesktop.setColor(QPalette::Base, QColor("#15191f"));
    darkDesktop.setColor(QPalette::Text, QColor("#f0f2f5"));
    darkDesktop.setColor(QPalette::Button, QColor("#20242b"));
    darkDesktop.setColor(QPalette::ButtonText, QColor("#f0f2f5"));
    QApplication::setPalette(darkDesktop);
    MainWindow window(app); // Window owns the C application.
    window.show();
    QTest::qWait(30);
    auto *scene = window.findChild<GraphScene *>();
    auto *view = window.findChild<GraphView *>();
    auto *palette = window.findChild<QTreeWidget *>("palette");
    auto *inspector = window.findChild<QTreeWidget *>("inspector");
    auto *diagnostics = window.findChild<QTreeWidget *>("diagnostics");
    QVERIFY(scene && view && palette && inspector && diagnostics);
    QVERIFY(window.findChild<QPushButton *>("createStereotypeButton"));
    QVERIFY(window.findChild<QPushButton *>("createDatasetButton"));
    QVERIFY(window.styleSheet().contains(QStringLiteral("selection-background-color: #3978c5")));
    QVERIFY(window.styleSheet().contains(QStringLiteral("color: #202c3b")));
    QCOMPARE(window.palette().color(QPalette::Window), QColor("#eceff3"));
    QVERIFY(window.styleSheet().contains(QStringLiteral("QPushButton { background: #e5e9ee; color: #202c3b;")));
    QVERIFY(palette->topLevelItemCount() > 0);
    QTreeWidgetItem *lightPalettePackage = nullptr;
    for (int group = 0; group < palette->topLevelItemCount(); ++group)
        for (int row = 0; row < palette->topLevelItem(group)->childCount(); ++row)
            if (palette->topLevelItem(group)->child(row)->text(0) == QStringLiteral("AdaptiveAvgPool2d"))
                lightPalettePackage = palette->topLevelItem(group)->child(row);
    QVERIFY(lightPalettePackage);
    QVERIFY(contrastAgainstWhite(lightPalettePackage->foreground(0).color()) >= 4.5);
    auto *resources = window.findChild<QTreeWidget *>("resources");
    QVERIFY(resources);
    QTreeWidgetItem *resourcePackage = nullptr;
    if (resources->topLevelItemCount() > 1)
        for (int row = 0; row < resources->topLevelItem(1)->childCount(); ++row)
            if (resources->topLevelItem(1)->child(row)->text(0) == QStringLiteral("core.adaptive-avg-pool2d@0.1.0"))
                resourcePackage = resources->topLevelItem(1)->child(row);
    QVERIFY(resourcePackage);
    QVERIFY(contrastAgainstWhite(resourcePackage->foreground(0).color()) >= 4.5);
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
    const QString smallCapture = qEnvironmentVariable("NN_TEST_CAPTURE_SMALL");
    if (!smallCapture.isEmpty()) {
        window.resize(window.minimumSize());
        QCoreApplication::processEvents();
        QVERIFY(window.grab().save(smallCapture));
    }

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
    QApplication::setPalette(systemPalette);
}

void WindowTest::visualResourceAuthoringForms() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "author-test", "Authoring test", false,
                           error, sizeof(error)), error);
    MainWindow window(app);

    auto fillDialog = [&window](const std::function<void(QDialog *)> &fill) {
        QTimer::singleShot(0, &window, [&window, fill] {
            auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            if (dialog) fill(dialog);
        });
    };
    auto acceptDialog = [](QDialog *dialog) {
        auto *buttons = dialog->findChild<QDialogButtonBox *>();
        if (buttons) buttons->button(QDialogButtonBox::Save)->click();
    };

    fillDialog([acceptDialog](QDialog *dialog) {
        dialog->findChild<QLineEdit *>("stereotypeId")->setText("ui-author");
        dialog->findChild<QLineEdit *>("stereotypeVersion")->setText("1.0.0");
        dialog->findChild<QLineEdit *>("stereotypeName")->setText("UI authoring fixture");
        QCOMPARE(dialog->findChild<QComboBox *>("stereotypeKind")->currentText(), QStringLiteral("layer"));
        QCOMPARE(dialog->findChild<QPlainTextEdit *>("stereotypeLua")->toPlainText(), QStringLiteral(
            "return function(context, parameters, services)\n"
            "  if not context.inputs[1] then\n"
            "    return { status = 'unresolved', message = 'Input missing' }\n"
            "  end\n"
            "  return { status = 'success', output = context.inputs[1] }\n"
            "end"));
        auto *parameters = dialog->findChild<QTableWidget *>("stereotypeParameters");
        for (QPushButton *button : dialog->findChildren<QPushButton *>())
            if (button->text() == QStringLiteral("Add parameter row")) {
                button->click();
                button->click();
                break;
            }
        parameters->item(0, 0)->setText("top_value");
        parameters->item(0, 1)->setText("integer");
        parameters->item(0, 2)->setText("9007199254740993");
        parameters->item(0, 5)->setText("top");
        parameters->item(1, 0)->setText("bottom_value");
        parameters->item(1, 1)->setText("string");
        parameters->item(1, 2)->setText("small");
        parameters->item(1, 5)->setText("bottom");
        acceptDialog(dialog);
    });
    QAction *stereotypeAction = nullptr;
    for (QAction *action : window.findChildren<QAction *>())
        if (action->text() == QString::fromUtf8("Create stereotype…")) stereotypeAction = action;
    QVERIFY(stereotypeAction);
    stereotypeAction->trigger();
    QVERIFY(nn_catalog_find(nn_project_catalog(nn_app_project(app)), "ui-author", "1.0.0"));
    auto *palette = window.findChild<QTreeWidget *>("palette");
    QVERIFY(palette);
    QTreeWidgetItem *authoredPackage = nullptr;
    for (int group = 0; group < palette->topLevelItemCount(); ++group) {
        auto *category = palette->topLevelItem(group);
        for (int row = 0; row < category->childCount(); ++row)
            if (category->child(row)->text(0) == QStringLiteral("UI authoring fixture"))
                authoredPackage = category->child(row);
    }
    QVERIFY(authoredPackage);
    palette->setCurrentItem(authoredPackage);
    auto *addNode = window.findChild<QPushButton *>("addNodeButton");
    QVERIFY(addNode);
    addNode->click();
    QCOMPARE(nn_model_node_count(nn_app_model(app)), size_t(1));
    auto *scene = window.findChild<GraphScene *>();
    QVERIFY(scene);
    NodeItem *card = nullptr;
    for (size_t i = 0; i < nn_model_node_count(nn_app_model(app)); ++i) {
        const NNNode *node = nn_model_node_at(nn_app_model(app), i);
        if (node && std::strcmp(node->package_id, "ui-author") == 0)
            card = scene->nodeItem(QString::fromUtf8(node->id));
    }
    QVERIFY(card);
    QCOMPARE(card->topParameters().size(), 1);
    QCOMPARE(card->bottomParameters().size(), 1);
    QCOMPARE(card->topParameters().first().second, QStringLiteral("9007199254740993"));
    PortItem *inputPort = nullptr, *outputPort = nullptr;
    for (PortItem *port : card->ports()) {
        if (port->isOutput()) outputPort = port;
        else inputPort = port;
    }
    QVERIFY(inputPort && outputPort);
    QCOMPARE(inputPort->y(), 9.0);
    QCOMPARE(outputPort->y(), card->boundingRect().height() + 1.0 - 9.0);
    const QString cardId = card->id();
    const QByteArray cardIdBytes = cardId.toUtf8();
    QVERIFY(nn_app_set_parameter_text(app, cardIdBytes.constData(), "top_value", "8",
                                      error, sizeof(error)));
    scene->refresh();
    card = scene->nodeItem(cardId);
    QCOMPARE(card->topParameters().first().second, QStringLiteral("8"));
    NNInferenceReport *withoutInput = nn_infer_project(nn_app_project(app));
    QVERIFY(withoutInput);
    bool authoredRuleUnresolved = false;
    for (size_t i = 0; i < nn_inference_count(withoutInput); ++i) {
        const NNInferenceResult *result = nn_inference_at(withoutInput, i);
        if (result && result->node_id && cardId == QString::fromUtf8(result->node_id))
            authoredRuleUnresolved = result->status == NN_INFERENCE_UNRESOLVED;
    }
    nn_inference_free(withoutInput);
    QVERIFY(authoredRuleUnresolved);

    fillDialog([acceptDialog](QDialog *dialog) {
        dialog->findChild<QLineEdit *>("datasetId")->setText("ui.dataset");
        dialog->findChild<QLineEdit *>("datasetVersion")->setText("1.0.0");
        dialog->findChild<QLineEdit *>("datasetName")->setText("UI dataset fixture");
        auto *inputs = dialog->findChild<QTableWidget *>("datasetInputs");
        auto *targets = dialog->findChild<QTableWidget *>("datasetTargets");
        for (QPushButton *button : dialog->findChildren<QPushButton *>())
            if (button->text() == QStringLiteral("Add row")) button->click();
        inputs->item(0, 0)->setText("features");
        inputs->item(0, 1)->setText("float32");
        inputs->item(0, 2)->setText("B, 28");
        targets->item(0, 0)->setText("label");
        targets->item(0, 1)->setText("int64");
        targets->item(0, 2)->setText("B");
        acceptDialog(dialog);
    });
    QAction *datasetAction = nullptr;
    for (QAction *action : window.findChildren<QAction *>())
        if (action->text() == QString::fromUtf8("Create dataset…")) datasetAction = action;
    QVERIFY(datasetAction);
    datasetAction->trigger();
    const NNProject *project = nn_app_project(app);
    QCOMPARE(nn_project_dataset_count(project), size_t(1));
    QCOMPARE(QString::fromUtf8(nn_project_active_dataset(project)->id), QStringLiteral("ui.dataset"));
    QVERIFY(nn_app_add_node(app, "source-input", "core.input", "0.1.0", "", 20, 40,
                            error, sizeof(error)));
    QVERIFY(nn_app_set_parameter_text(app, "source-input", "binding", "features",
                                      error, sizeof(error)));
    QVERIFY(nn_app_connect(app, "authored-edge", "source-input", "out", cardIdBytes.constData(),
                           "in", error, sizeof(error)));
    NNInferenceReport *report = nn_infer_project(nn_app_project(app));
    QVERIFY(report);
    bool authoredRuleSucceeded = false;
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (result && result->node_id && cardId == QString::fromUtf8(result->node_id))
            authoredRuleSucceeded = result->status == NN_INFERENCE_SUCCESS;
    }
    nn_inference_free(report);
    QVERIFY(authoredRuleSucceeded);
    QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
    const QByteArray projectPath = QByteArray(temporary.path().toUtf8()) + "/author-test";
    NNApplication *reopened = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(reopened);
    QVERIFY2(nn_app_open(reopened, projectPath.constData(), error, sizeof(error)), error);
    QCOMPARE(nn_project_dataset_count(nn_app_project(reopened)), size_t(1));
    QCOMPARE(QString::fromUtf8(nn_project_active_dataset(nn_app_project(reopened))->id), QStringLiteral("ui.dataset"));
    nn_app_free(reopened);
    window.close();
}

void WindowTest::automationRejectsInvalidScopeAndCapturesCurrentScope() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "scope-test", "Scope test", false,
                           error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "flow", "core.subflow-proxy", "0.1.0", "", 100, 100,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "child", "core.relu", "0.1.0", "flow", 100, 100,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "encoder", "core.subflow-proxy", "0.1.0", "", 400, 100,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "encoder-child", "core.relu", "0.1.0", "encoder", 100, 100,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
    MainWindow window(app);
    const QString socketPath = temporary.path() + QStringLiteral("/automation.sock");
    QVERIFY2(window.startAutomation(socketPath), "Could not start UI automation service");

    auto runCli = [](const QStringList &arguments) {
        QProcess process;
        process.start(QCoreApplication::applicationDirPath() + QStringLiteral("/nnmodelctl"), arguments);
        if (!process.waitForStarted(1500)) return QPair<int, QByteArray>(-1, process.errorString().toUtf8());
        QEventLoop loop;
        QObject::connect(&process, &QProcess::finished, &loop, &QEventLoop::quit);
        QTimer timeout;
        timeout.setSingleShot(true);
        QObject::connect(&timeout, &QTimer::timeout, &loop, &QEventLoop::quit);
        timeout.start(5000);
        if (process.state() != QProcess::NotRunning) loop.exec();
        if (process.state() != QProcess::NotRunning) {
            process.kill();
            process.waitForFinished(1000);
            return QPair<int, QByteArray>(-2, QByteArray("CLI timed out"));
        }
        return QPair<int, QByteArray>(process.exitCode(), process.readAllStandardOutput());
    };
    auto inspectScope = [&runCli, &socketPath]() {
        const auto response = runCli({"--socket", socketPath, "ui.inspect"});
        if (response.first != 0) return QStringLiteral("<inspect-failed>");
        const QJsonObject result = QJsonDocument::fromJson(response.second).object()
                                       .value(QStringLiteral("result")).toObject();
        return result.value(QStringLiteral("currentScope")).toString();
    };

    const auto missingScope = runCli({"--socket", socketPath, "ui.scope"});
    QVERIFY(missingScope.first != 0);
    QCOMPARE(inspectScope(), QString());
    const auto nonStringScope = runCli({"--socket", socketPath, "ui.scope", "{\"id\":42}"});
    QVERIFY(nonStringScope.first != 0);
    QCOMPARE(inspectScope(), QString());
    const auto unknownScope = runCli({"--socket", socketPath, "ui.scope", "{\"id\":\"missing\"}"});
    QVERIFY(unknownScope.first != 0);
    QCOMPARE(inspectScope(), QString());

    const auto entered = runCli({"--socket", socketPath, "ui.scope", "{\"id\":\"flow\"}"});
    QCOMPARE(entered.first, 0);
    QCOMPARE(inspectScope(), QStringLiteral("flow"));
    const QString screenshot = temporary.path() + QStringLiteral("/scope.png");
    const QByteArray screenshotArgs = QJsonDocument(QJsonObject{{QStringLiteral("path"), screenshot}})
                                         .toJson(QJsonDocument::Compact);
    const auto captured = runCli({"--socket", socketPath, "ui.screenshot", QString::fromUtf8(screenshotArgs)});
    QCOMPARE(captured.first, 0);
    QVERIFY(QFileInfo::exists(screenshot));
    QCOMPARE(inspectScope(), QStringLiteral("flow"));

    const auto enterEncoder = runCli({"--socket", socketPath, "ui.scope", "{\"id\":\"encoder\"}"});
    QCOMPARE(enterEncoder.first, 0);
    QCOMPARE(inspectScope(), QStringLiteral("encoder"));
    const QByteArray vaePath = QByteArray(NN_SOURCE_DIR) + "/examples/mnist-vae";
    const QByteArray openArgs = QJsonDocument(QJsonObject{{QStringLiteral("path"), QString::fromUtf8(vaePath)}})
                                    .toJson(QJsonDocument::Compact);
    const auto opened = runCli({"--socket", socketPath, "project.open", QString::fromUtf8(openArgs)});
    QCOMPARE(opened.first, 0);
    QCOMPARE(inspectScope(), QString());
    QCOMPARE(QString::fromUtf8(nn_project_id(nn_app_project(app))), QStringLiteral("mnist-vae"));
}

void WindowTest::socketOptionRequiresPath() {
    QProcess process;
    process.setProgram(QCoreApplication::applicationDirPath() + QStringLiteral("/nnmodelling-ui"));
    process.setArguments({QStringLiteral("--socket")});
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.insert(QStringLiteral("QT_QPA_PLATFORM"), QStringLiteral("offscreen"));
    process.setProcessEnvironment(environment);
    process.start();
    QVERIFY(process.waitForStarted(2000));
    QVERIFY(process.waitForFinished(5000));
    QCOMPARE(process.exitCode(), 2);
    QVERIFY(process.readAllStandardError().contains("--socket requires a socket path"));
}

QTEST_MAIN(WindowTest)
#include "qt_window_test.moc"
