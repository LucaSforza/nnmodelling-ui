#include "MainWindow.hpp"
#include "ActivityLog.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "MainWindowUtils.hpp"
#include "PortItem.hpp"
#include "EdgeItem.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "inference/inference.h"
#include "model/model.h"
#include "project/project.h"

#include <QAction>
#include <QApplication>
#include <QDir>
#include <QDirIterator>
#include <QJsonArray>
#include <QLayout>
#include <QImage>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>
#include <QComboBox>
#include <QPushButton>
#include <QMessageBox>
#include <QMap>
#include <QMenu>
#include <QMenuBar>
#include <QLineF>
#include <QDialog>
#include <QDialogButtonBox>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QPlainTextEdit>
#include <QPalette>
#include <QScrollArea>
#include <QTabBar>
#include <QTabWidget>
#include <cmath>
#include <QAbstractButton>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <QTreeWidget>
#include <QTreeWidgetItemIterator>
#include <QListWidget>
#include <QToolBar>
#include <QToolButton>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

class WindowTest : public QObject {
    Q_OBJECT
private slots:
    void lifecycleAndInspector();
    void appDialogsContrastUnderDarkPalette();
    void miniLlmTemplateShortcut();
    void componentGuideAppearsBelowAddButton();
    void startupChooserShowsEditorOnlyAfterProjectCreation();
    void stereotypeReferenceForm();
    void modelProblemDiagnosticsAndNavigation();
    void toolbarDeleteTracksSelection();
    void currentScopeRetainsOutsideCauseContext();
    void retainedLuaValidationFormAndNonblockingRejection();
    void visualResourceAuthoringForms();
    void datasetManagerEditFormAndRetainedValidation();
    void datasetImportFromEmptyDirectory();
    void automaticDatasetBindingInspectorAndActivation();
    void typedOutputAuthoringBoundariesAndInspector();
    void automationRejectsInvalidScopeAndCapturesCurrentScope();
    void arrangeMenuAndFitUseDirectedGridLayout();
    void graphHistoryTracksEditsAndScopeFallback();
    void scopeTreeFollowsContainmentAndMenus();
    void socketOptionRequiresPath();
    void activityLogRecordsUiAndCoreEvents();
};

void WindowTest::miniLlmTemplateShortcut() {
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    MainWindow window(app);
    bool foundMlp = false;
    bool foundVae = false;
    bool foundTinyLlm = false;
    for (QAction *action : window.findChildren<QAction *>()) {
        if (action->text() == "MNIST MLP…") foundMlp = true;
        if (action->text() == "MNIST VAE…") foundVae = true;
        if (action->text() == "tiny LLM…") foundTinyLlm = true;
    }
    QVERIFY(foundMlp);
    QVERIFY(foundVae);
    QVERIFY(foundTinyLlm);
    bool foundCreate = false;
    bool foundOpen = false;
    QTimer::singleShot(0, &window, [&] {
        auto *chooser = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        if (!chooser) return;
        for (QAbstractButton *button : chooser->buttons()) {
            if (button->text() == "Create project") foundCreate = true;
            if (button->text() == "Open project") foundOpen = true;
        }
        chooser->button(QMessageBox::Close)->click();
    });
    window.showProjectChooser();
    QVERIFY(foundCreate);
    QVERIFY(foundOpen);
}

void WindowTest::componentGuideAppearsBelowAddButton() {
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    MainWindow window(app);
    auto *guide = window.findChild<QLabel *>("componentGuide");
    auto *addButton = window.findChild<QPushButton *>("addNodeButton");
    QVERIFY(guide && addButton);
    QVERIFY(!addButton->isEnabled());
    QCOMPARE(guide->parentWidget(), addButton->parentWidget());
    QLayout *layout = guide->parentWidget()->layout();
    QVERIFY(layout);
    QVERIFY(layout->indexOf(addButton) < layout->indexOf(guide));
}

void WindowTest::startupChooserShowsEditorOnlyAfterProjectCreation() {
    struct NativeDialogSetting {
        bool previous = QCoreApplication::testAttribute(Qt::AA_DontUseNativeDialogs);
        NativeDialogSetting() { QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, true); }
        ~NativeDialogSetting() { QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, previous); }
    } nativeDialogSetting;
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    MainWindow window(app);
    const bool originalQuitPolicy = QApplication::quitOnLastWindowClosed();
    bool editorHiddenAtChooser = false;
    bool creationAccepted = false;
    bool quitSuppressedAtChooser = false;
    bool quitSuppressedAtCreation = false;

    QTimer::singleShot(0, &window, [&] {
        auto *chooser = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        QVERIFY(chooser);
        editorHiddenAtChooser = !window.isVisible();
        quitSuppressedAtChooser = !QApplication::quitOnLastWindowClosed();
        QTimer::singleShot(0, &window, [&] {
            auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            QVERIFY(dialog);
            quitSuppressedAtCreation = !QApplication::quitOnLastWindowClosed();
            auto *parent = dialog->findChild<QLineEdit *>("projectParent");
            QVERIFY(parent);
            parent->setText(temporary.path());
            for (QPushButton *button : dialog->findChildren<QPushButton *>()) {
                if (button->text() != QStringLiteral("Create")) continue;
                creationAccepted = true;
                button->click();
                return;
            }
            QFAIL("Create button missing from project form");
        });
        for (QAbstractButton *button : chooser->buttons()) {
            if (button->text() != QStringLiteral("Create project")) continue;
            button->click();
            return;
        }
        QFAIL("Create project button missing from startup chooser");
    });

    window.showProjectChooser(true);
    QVERIFY(editorHiddenAtChooser);
    QVERIFY(creationAccepted);
    QVERIFY(quitSuppressedAtChooser);
    QVERIFY(quitSuppressedAtCreation);
    QVERIFY(nn_app_project(app));
    QVERIFY(window.isVisible());
    QCOMPARE(QApplication::quitOnLastWindowClosed(), originalQuitPolicy);
    QCoreApplication::processEvents();
    auto *view = window.findChild<GraphView *>();
    auto *scene = window.findChild<GraphScene *>();
    QVERIFY(view && scene);
    const QRectF visibleBounds = view->mapToScene(view->viewport()->rect()).boundingRect();
    QVERIFY2(visibleBounds.contains(scene->contentBounds()),
        "Startup project graph should fit after editor window is shown");

    bool quitSuppressedAtOpenChooser = false;
    bool quitSuppressedAtPicker = false;
    QTimer::singleShot(0, &window, [&] {
        auto *chooser = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        QVERIFY(chooser);
        quitSuppressedAtOpenChooser = !QApplication::quitOnLastWindowClosed();
        QTimer::singleShot(0, &window, [&] {
            auto *picker = qobject_cast<QFileDialog *>(QApplication::activeModalWidget());
            QVERIFY(picker);
            quitSuppressedAtPicker = !QApplication::quitOnLastWindowClosed();
            picker->setDirectory(temporary.path());
            picker->selectFile(QStringLiteral("mnist-mlp"));
            QVERIFY(QMetaObject::invokeMethod(picker, "accept", Qt::DirectConnection));
        });
        for (QAbstractButton *button : chooser->buttons()) {
            if (button->text() != QStringLiteral("Open project")) continue;
            button->click();
            return;
        }
        QFAIL("Open project button missing from startup chooser");
    });
    window.showProjectChooser(true);
    QVERIFY(quitSuppressedAtOpenChooser);
    QVERIFY(quitSuppressedAtPicker);
    QVERIFY(window.isVisible());
    QCOMPARE(QString::fromUtf8(nn_project_id(nn_app_project(app))), QStringLiteral("mnist-mlp"));
    QCOMPARE(QApplication::quitOnLastWindowClosed(), originalQuitPolicy);
}

void WindowTest::stereotypeReferenceForm() {
    QTemporaryDir temporary;
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    QVERIFY2(nn_app_create(app, temporary.path().toUtf8().constData(), "reference-form",
                          "Reference form", "0.1.0", false, error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "parallel", "core.horizontal-repeat", "0.1.0", "",
                            0, 0, error, sizeof(error)), error);
    QVERIFY2(nn_app_create_stereotype(app, "form.join", "1.0.0",
        R"({"name":"Form join","kind":"join","view":{"color":"#4779c4","width":180,"height":100},"parameters":{"enabled":{"type":"boolean","default":true},"factor":{"type":"number","default":1.0,"minimum":0},"mode":{"type":"string","default":"sum","choices":["sum","mean","$literal"]}}})",
        "return function(c,p,s) return {status='success',output=c.inputs[1]} end",
        "{}", error, sizeof(error)), error);
    const QByteArray variables = QByteArrayLiteral("[{\"name\":\"factor\",\"type\":\"number\",\"value\":3.5},{\"name\":\"enabledFlag\",\"type\":\"boolean\",\"value\":true},{\"name\":\"modeName\",\"type\":\"string\",\"value\":\"sum\"}]");
    QVERIFY2(nn_app_set_metavariables_json(app, variables.constData(), error, sizeof(error)), error);
    MainWindow window(app);
    window.show();
    auto *scene = window.findChild<GraphScene *>();
    QVERIFY(scene);
    scene->nodeItem("parallel")->setSelected(true);
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    auto *selector = window.findChild<QComboBox *>("stereotypeReference_join");
    auto *dimension = window.findChild<QLineEdit *>("stereotypeParameter_join_dim");
    QVERIFY(selector && dimension);
    QVERIFY(!selector->currentText().startsWith(QLatin1Char('{')));
    QCOMPARE(dimension->text(), QStringLiteral("-1"));
    QVERIFY(!dimension->validator());
    QVERIFY(!window.findChild<QLineEdit *>("stereotypeParameters_join"));
    dimension->setText("0");
    QVERIFY(QMetaObject::invokeMethod(dimension, "editingFinished", Qt::DirectConnection));
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    char *reference = nn_app_parameter_text(app, "parallel", "join");
    QVERIFY(reference);
    QCOMPARE(QJsonDocument::fromJson(reference).object().value("parameters").toObject().value("dim").toInt(), 0);
    nn_app_free_text(reference);
    selector = window.findChild<QComboBox *>("stereotypeReference_join");
    int add = -1;
    for (int i = 1; i < selector->count(); ++i) {
        const QJsonObject item = selector->itemData(i).toJsonObject();
        QVERIFY(item.value("id").toString() != QStringLiteral("core.relu"));
        if (item.value("id").toString() == QStringLiteral("core.add")) add = i;
    }
    QVERIFY(add > 0);
    selector->setCurrentIndex(add);
    QVERIFY(QMetaObject::invokeMethod(selector, "activated", Qt::DirectConnection, Q_ARG(int, add)));
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    QVERIFY(!window.findChild<QLineEdit *>("stereotypeParameter_join_dim"));
    QVERIFY2(nn_app_undo(app, error, sizeof(error)), error);
    scene->refresh();
    scene->nodeItem("parallel")->setSelected(false);
    scene->nodeItem("parallel")->setSelected(true);
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    dimension = window.findChild<QLineEdit *>("stereotypeParameter_join_dim");
    QVERIFY(dimension);
    QCOMPARE(dimension->text(), QStringLiteral("0"));
    selector = window.findChild<QComboBox *>("stereotypeReference_join");
    int custom = -1;
    for (int i = 1; i < selector->count(); ++i)
        if (selector->itemData(i).toJsonObject().value("id").toString() == QStringLiteral("form.join")) custom = i;
    QVERIFY(custom > 0);
    selector->setCurrentIndex(custom);
    QVERIFY(QMetaObject::invokeMethod(selector, "activated", Qt::DirectConnection, Q_ARG(int, custom)));
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    auto *enabled = window.findChild<QComboBox *>("stereotypeParameter_join_enabled");
    auto *factor = window.findChild<QLineEdit *>("stereotypeParameter_join_factor");
    auto *mode = window.findChild<QComboBox *>("stereotypeParameter_join_mode");
    QVERIFY(enabled && factor && mode);
    QVERIFY(enabled->isEditable());
    QCOMPARE(enabled->currentText(), QStringLiteral("true"));
    QCOMPARE(mode->count(), 3);
    factor->setText("2.5");
    QVERIFY(QMetaObject::invokeMethod(factor, "editingFinished", Qt::DirectConnection));
    enabled->setEditText(QStringLiteral("false"));
    QVERIFY(QMetaObject::invokeMethod(enabled->lineEdit(), "editingFinished", Qt::DirectConnection));
    mode->setCurrentText("mean");
    QVERIFY(QMetaObject::invokeMethod(mode->lineEdit(), "editingFinished", Qt::DirectConnection));
    reference = nn_app_parameter_text(app, "parallel", "join");
    const QJsonObject values = QJsonDocument::fromJson(reference).object().value("parameters").toObject();
    nn_app_free_text(reference);
    QCOMPARE(values.value("factor").toDouble(), 2.5);
    QCOMPARE(values.value("enabled").toBool(), false);
    QCOMPARE(values.value("mode").toString(), QStringLiteral("mean"));
    bool invalidChoiceWasRejected = false;
    QTimer::singleShot(0, &window, [&] {
        auto *dialog = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        if (dialog) {
            invalidChoiceWasRejected = true;
            dialog->accept();
        }
    });
    mode->setEditText(QStringLiteral("not-a-choice"));
    QVERIFY(QMetaObject::invokeMethod(mode->lineEdit(), "editingFinished", Qt::DirectConnection));
    QVERIFY(invalidChoiceWasRejected);
    reference = nn_app_parameter_text(app, "parallel", "join");
    QCOMPARE(QJsonDocument::fromJson(reference).object().value("parameters").toObject()
        .value("mode").toString(), QStringLiteral("mean"));
    nn_app_free_text(reference);
    mode = window.findChild<QComboBox *>("stereotypeParameter_join_mode");
    QVERIFY(mode);
    const int dollarChoice = mode->findText(QStringLiteral("$literal"), Qt::MatchExactly);
    QVERIFY(dollarChoice > 0);
    mode->setCurrentIndex(dollarChoice);
    QVERIFY(QMetaObject::invokeMethod(mode, "activated", Qt::DirectConnection, Q_ARG(int, dollarChoice)));
    reference = nn_app_parameter_text(app, "parallel", "join");
    QCOMPARE(QJsonDocument::fromJson(reference).object().value("parameters").toObject()
        .value("mode").toString(), QStringLiteral("$literal"));
    nn_app_free_text(reference);
    enabled->setEditText(QStringLiteral("$enabledFlag"));
    QVERIFY(QMetaObject::invokeMethod(enabled->lineEdit(), "editingFinished", Qt::DirectConnection));
    mode->setEditText(QStringLiteral("$modeName"));
    QVERIFY(QMetaObject::invokeMethod(mode->lineEdit(), "editingFinished", Qt::DirectConnection));
    reference = nn_app_parameter_text(app, "parallel", "join");
    const QJsonObject boundFields = QJsonDocument::fromJson(reference).object()
        .value("parameters").toObject();
    nn_app_free_text(reference);
    QCOMPARE(boundFields.value("enabled").toObject().value("$ref").toString(), QStringLiteral("enabledFlag"));
    QCOMPARE(boundFields.value("mode").toObject().value("$ref").toString(), QStringLiteral("modeName"));
    QCOMPARE(enabled->currentText(), QStringLiteral("$enabledFlag"));
    QCOMPARE(mode->currentText(), QStringLiteral("$modeName"));
    QVERIFY(!window.findChild<QLineEdit *>(QStringLiteral("stereotypeParameter_join_factor_binding")));
    factor->setText(QStringLiteral("$factor"));
    QVERIFY(QMetaObject::invokeMethod(factor, "editingFinished", Qt::DirectConnection));
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    reference = nn_app_parameter_text(app, "parallel", "join");
    const QJsonValue factorSource = QJsonDocument::fromJson(reference).object()
        .value("parameters").toObject().value("factor");
    nn_app_free_text(reference);
    QCOMPARE(factorSource.toObject().value(QStringLiteral("$ref")).toString(), QStringLiteral("factor"));
    factor = window.findChild<QLineEdit *>("stereotypeParameter_join_factor");
    QVERIFY(factor);
    QCOMPARE(factor->text(), QStringLiteral("$factor"));
    auto *inspector = window.findChild<QTreeWidget *>(QStringLiteral("inspector"));
    QVERIFY(inspector);
    bool foundResolvedFactor = false;
    QTreeWidgetItem *factorRow = nullptr;
    QTreeWidgetItem *factorResolvedRow = nullptr;
    for (int i = 0; i < inspector->topLevelItemCount(); ++i) {
        QTreeWidgetItem *top = inspector->topLevelItem(i);
        if (top->text(0) != QStringLiteral("Parameters")) continue;
        for (int j = 0; j < top->childCount(); ++j) {
            QTreeWidgetItem *parameter = top->child(j);
            if (parameter->text(0) != QStringLiteral("join")) continue;
            for (int k = 0; k < parameter->childCount(); ++k) {
                QTreeWidgetItem *field = parameter->child(k);
                if (field->text(0) != QStringLiteral("factor")) continue;
                for (int n = 0; n < field->childCount(); ++n) {
                    QTreeWidgetItem *resolved = field->child(n);
                    if (resolved->text(0) == QStringLiteral("Resolved") &&
                        resolved->text(1).contains(QStringLiteral("3.5"))) {
                        foundResolvedFactor = true;
                        factorRow = field;
                        factorResolvedRow = resolved;
                    }
                }
            }
        }
    }
    QVERIFY(foundResolvedFactor);
    QVERIFY(factorRow && factorResolvedRow);
    window.resize(700, 520);
    QFont enlarged = inspector->font();
    enlarged.setPointSize(enlarged.pointSize() + 5);
    inspector->setFont(enlarged);
    QCoreApplication::processEvents();
    const QRect factorRect = inspector->visualItemRect(factorRow);
    const QRect resolvedRect = inspector->visualItemRect(factorResolvedRow);
    QVERIFY(factorRect.isValid() && resolvedRect.isValid());
    QVERIFY(resolvedRect.top() >= factorRect.bottom());
    QWidget *factorEditor = inspector->itemWidget(factorRow, 1);
    QVERIFY(factorEditor);
    QVERIFY(factorRect.contains(factorEditor->geometry()));
    QVERIFY2(nn_app_undo(app, error, sizeof(error)), error);
    reference = nn_app_parameter_text(app, "parallel", "join");
    QVERIFY(reference);
    QCOMPARE(QJsonDocument::fromJson(reference).object().value("parameters").toObject()
        .value("factor").toDouble(), 2.5);
    nn_app_free_text(reference);
    QVERIFY2(nn_app_redo(app, error, sizeof(error)), error);
    reference = nn_app_parameter_text(app, "parallel", "join");
    QVERIFY(reference);
    QCOMPARE(QJsonDocument::fromJson(reference).object().value("parameters").toObject()
        .value("factor").toObject().value("$ref").toString(), QStringLiteral("factor"));
    nn_app_free_text(reference);
    QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
    window.close();
}

void WindowTest::arrangeMenuAndFitUseDirectedGridLayout() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "layout-test", "Layout test", "0.1.0", false,
                           error, sizeof(error)), error);
    auto add = [app, &error](const char *id, const char *package) {
        return nn_app_add_node(app, id, package, "0.1.0", "", 0, 0, error, sizeof(error));
    };
    QVERIFY2(add("source", "core.input"), error);
    QVERIFY2(add("branch-a", "core.relu"), error);
    QVERIFY2(add("branch-b", "core.relu"), error);
    QVERIFY2(add("tail", "core.relu"), error);
    for (int i = 1; i <= 6; ++i) {
        const QByteArray id = QStringLiteral("chain-%1").arg(i).toUtf8();
        QVERIFY2(add(id.constData(), "core.relu"), error);
    }
    const NNModel *model = nn_app_model(app);
    const NNNode *output = nullptr;
    const NNNode *lossOutput = nullptr;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node && node->package_id && std::strcmp(node->package_id, "core.output") == 0)
            output = node;
        if (node && node->package_id && std::strcmp(node->package_id, "core.loss-output") == 0)
            lossOutput = node;
    }
    QVERIFY(output && lossOutput);
    const QByteArray outputId(output->id);
    const QByteArray lossOutputId(lossOutput->id);
    QVERIFY2(nn_app_connect(app, "branch-a-edge", "source", "out", "branch-a", "in",
                            error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "branch-b-edge", "source", "out", "branch-b", "in",
                            error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "tail-edge", "branch-a", "out", "tail", "in",
                            error, sizeof(error)), error);
    QByteArray previous = "tail";
    for (int i = 1; i <= 6; ++i) {
        const QByteArray target = QStringLiteral("chain-%1").arg(i).toUtf8();
        const QByteArray edge = QStringLiteral("chain-edge-%1").arg(i).toUtf8();
        QVERIFY2(nn_app_connect(app, edge.constData(), previous.constData(), "out",
                                target.constData(), "in", error, sizeof(error)), error);
        previous = target;
    }
    QVERIFY2(nn_app_connect(app, "output-edge", previous.constData(), "out", outputId.constData(), "in",
                            error, sizeof(error)), error);

    MainWindow window(app);
    window.show();
    QCoreApplication::processEvents();
    auto *scene = window.findChild<GraphScene *>();
    auto *view = window.findChild<GraphView *>();
    QVERIFY(scene && view);
    auto verifyLayoutClearance = [&] {
        model = nn_app_model(app);
        std::vector<const NodeItem *> items;
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node || (node->scope_id && *node->scope_id)) continue;
            const NodeItem *item = scene->nodeItem(QString::fromUtf8(node->id));
            if (item) items.push_back(item);
        }
        for (size_t i = 0; i < items.size(); ++i)
            for (size_t j = i + 1; j < items.size(); ++j)
                QVERIFY(!items[i]->sceneBoundingRect().intersects(items[j]->sceneBoundingRect()));
        for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
            const NNEdge *edge = nn_model_edge_at(model, i);
            if (!edge || (edge->scope_id && *edge->scope_id)) continue;
            const NodeItem *sourceItem = scene->nodeItem(QString::fromUtf8(edge->source_id));
            const NodeItem *targetItem = scene->nodeItem(QString::fromUtf8(edge->target_id));
            QVERIFY(sourceItem && targetItem);
            const QRectF sourceRect = sourceItem->sceneBoundingRect();
            const QRectF targetRect = targetItem->sceneBoundingRect();
            const qreal corridor = scene->flowDirection() == FlowDirection::Vertical
                ? targetRect.top() - sourceRect.bottom()
                : sourceRect.left() - targetRect.right();
            QVERIFY(corridor >= 60.0);
        }
    };
    QAction *zoomIn = window.findChild<QAction *>("zoomIn");
    QAction *zoomOut = window.findChild<QAction *>("zoomOut");
    QVERIFY(zoomIn && zoomOut);
    const QPoint zoomAnchor = view->viewport()->rect().center();
    const QPointF beforeZoom = view->mapToScene(zoomAnchor);
    zoomIn->trigger();
    QCOMPARE(view->transform().m11(), 1.15);
    QVERIFY(QLineF(beforeZoom, view->mapToScene(zoomAnchor)).length() < 2.0);
    zoomOut->trigger();
    QVERIFY(qAbs(view->transform().m11() - 1.0) < 0.001);
    QMenu *arrangeMenu = nullptr;
    for (QMenu *menu : window.findChildren<QMenu *>())
        if (menu->actions().size() == 2 && menu->actions()[0]->text() == QStringLiteral("Vertical"))
            arrangeMenu = menu;
    QVERIFY(arrangeMenu);
    arrangeMenu->actions()[0]->trigger();
    QCoreApplication::processEvents();
    QVERIFY(scene->flowDirection() == FlowDirection::Vertical);
    model = nn_app_model(app);
    const NNNode *source = nn_model_find_node(model, "source");
    const NNNode *branchA = nn_model_find_node(model, "branch-a");
    const NNNode *branchB = nn_model_find_node(model, "branch-b");
    const NNNode *tail = nn_model_find_node(model, "tail");
    const NNNode *chainEnd = nn_model_find_node(model, "chain-6");
    const NNNode *laidOutOutput = nn_model_find_node(model, outputId.constData());
    const NNNode *laidOutLoss = nn_model_find_node(model, lossOutputId.constData());
    QVERIFY(source && branchA && branchB && tail && chainEnd && laidOutOutput && laidOutLoss);
    QVERIFY(source->y < branchA->y && branchA->y < tail->y && tail->y < chainEnd->y &&
            chainEnd->y < laidOutOutput->y && chainEnd->y < laidOutLoss->y);
    QVERIFY(branchA->x != branchB->x);
    verifyLayoutClearance();
    for (const NNNode *node : {source, branchA, branchB, tail, chainEnd, laidOutOutput, laidOutLoss}) {
        QCOMPARE(qint32(node->x) % 20, 0);
        QCOMPARE(qint32(node->y) % 20, 0);
    }

    arrangeMenu->actions()[1]->trigger();
    QCoreApplication::processEvents();
    QVERIFY(scene->flowDirection() == FlowDirection::Horizontal);
    model = nn_app_model(app);
    source = nn_model_find_node(model, "source");
    branchA = nn_model_find_node(model, "branch-a");
    branchB = nn_model_find_node(model, "branch-b");
    tail = nn_model_find_node(model, "tail");
    chainEnd = nn_model_find_node(model, "chain-6");
    laidOutOutput = nn_model_find_node(model, outputId.constData());
    laidOutLoss = nn_model_find_node(model, lossOutputId.constData());
    QVERIFY(source && branchA && branchB && tail && chainEnd && laidOutOutput && laidOutLoss);
    QVERIFY(source->x > branchA->x && branchA->x > tail->x && tail->x > chainEnd->x &&
            chainEnd->x > laidOutOutput->x && chainEnd->x > laidOutLoss->x);
    QVERIFY(branchA->y != branchB->y);
    verifyLayoutClearance();
    QVERIFY(QLineF(view->mapFromScene(scene->itemsBoundingRect().center()),
                   view->viewport()->rect().center()).length() < 2.0);

    std::map<QByteArray, QPoint> beforeFit;
    model = nn_app_model(app);
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        beforeFit.emplace(QByteArray(node->id), QPoint(node->x, node->y));
    }
    const FlowDirection directionBeforeFit = scene->flowDirection();
    QAction *fit = nullptr;
    for (QAction *action : window.findChildren<QAction *>())
        if (action->objectName() == QStringLiteral("fitGraph")) fit = action;
    QVERIFY(fit);
    fit->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(scene->flowDirection(), directionBeforeFit);
    model = nn_app_model(app);
    for (const auto &entry : beforeFit) {
        const NNNode *node = nn_model_find_node(model, entry.first.constData());
        QVERIFY(node);
        QCOMPARE(QPoint(node->x, node->y), entry.second);
    }
    const QRectF focusedSceneRect = scene->sceneRect();
    const QRectF focusedItems = scene->itemsBoundingRect();
    QVERIFY2(focusedSceneRect.contains(focusedItems),
        qPrintable(QStringLiteral("scene rect %1,%2 %3x%4 excludes items %5,%6 %7x%8")
            .arg(focusedSceneRect.x()).arg(focusedSceneRect.y())
            .arg(focusedSceneRect.width()).arg(focusedSceneRect.height())
            .arg(focusedItems.x()).arg(focusedItems.y())
            .arg(focusedItems.width()).arg(focusedItems.height())));
    const QByteArray projectPath = (temporary.path() + QStringLiteral("/layout-test")).toUtf8();
    QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
    scene->setFlowDirection(FlowDirection::Horizontal);
    QVERIFY2(window.openProject(QString::fromUtf8(projectPath)), "Could not reopen test project");
    QVERIFY(scene->flowDirection() == FlowDirection::Vertical);
}

void WindowTest::scopeTreeFollowsContainmentAndMenus() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "scope-tree-test", "Scope tree", "0.1.0", false,
                           error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "outer", "core.subflow-proxy", "0.1.0", "", 0, 0,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "inner", "core.subflow-proxy", "0.1.0", "outer", 0, 0,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_rename_node(app, "outer", "Outer flow", error, sizeof(error)), error);

    MainWindow window(app);
    window.show();
    QCoreApplication::processEvents();
    auto *scene = window.findChild<GraphScene *>();
    auto *scopeButton = window.findChild<QToolButton *>("scopeSelector");
    auto *tree = window.findChild<QTreeWidget *>("scopeTree");
    QVERIFY(scene && scopeButton && tree);
    QCOMPARE(tree->topLevelItemCount(), 1);
    QTreeWidgetItem *root = tree->topLevelItem(0);
    QCOMPARE(root->text(0), QStringLiteral("Root"));
    QCOMPARE(root->childCount(), 1);
    QTreeWidgetItem *outer = root->child(0);
    QCOMPARE(outer->text(0), QStringLiteral("Outer flow"));
    QCOMPARE(outer->childCount(), 1);
    QCOMPARE(outer->child(0)->text(0), QStringLiteral("Subflow Proxy"));

    const QStringList menuNames = {window.menuBar()->actions()[0]->text(),
                                   window.menuBar()->actions()[1]->text(),
                                   window.menuBar()->actions()[2]->text(),
                                   window.menuBar()->actions()[3]->text(),
                                   window.menuBar()->actions()[4]->text()};
    QCOMPARE(menuNames, QStringList({QStringLiteral("&File"), QStringLiteral("&Edit"), QStringLiteral("&Model"),
                                     QStringLiteral("&Dataset"), QStringLiteral("&View")}));
    QMenu *modelMenu = window.menuBar()->actions()[2]->menu();
    QMenu *datasetMenu = window.menuBar()->actions()[3]->menu();
    QStringList datasetActions;
    for (QAction *action : datasetMenu->actions()) datasetActions << action->text();
    QCOMPARE(datasetActions, QStringList({QStringLiteral("Create dataset…"), QStringLiteral("Manage datasets…")}));
    for (QAction *action : modelMenu->actions()) QVERIFY(!datasetActions.contains(action->text()));
    QMenu *templates = window.findChild<QMenu *>("projectTemplatesMenu");
    QVERIFY(templates && templates->actions().size() == 3);
    QAction *open = window.findChild<QAction *>("openProjectAction");
    QAction *save = window.findChild<QAction *>("saveProjectAction");
    QVERIFY(open && save);
    QCOMPARE(open->shortcut(), QKeySequence::Open);
    QCOMPARE(save->shortcut(), QKeySequence::Save);
    for (QToolBar *toolbar : window.findChildren<QToolBar *>()) {
        for (QAction *action : toolbar->actions())
            QVERIFY(action->text() != QStringLiteral("Open project…") &&
                    action->text() != QStringLiteral("Save"));
    }

    // VNC renews popup containers after close. Actions and submenu window
    // flags must survive; losing Qt::Popup embeds the submenu in its parent.
    for (int i = 0; i < window.menuBar()->actions().size(); ++i) {
        const auto entries = window.menuBar()->actions()[i]->menu()->actions();
        for (int repeat = 0; repeat < 2; ++repeat) {
            QMenu *menu = window.menuBar()->actions()[i]->menu();
            QVERIFY(menu->windowFlags().testFlag(Qt::Popup));
            QCOMPARE(menu->actions(), entries);
            menu->popup(window.mapToGlobal(QPoint(20, 40)));
            QCoreApplication::processEvents();
            QVERIFY(menu->isVisible());
            menu->close();
            QTest::qWait(25);
        }
    }
    templates = window.findChild<QMenu *>("projectTemplatesMenu");
    QVERIFY(templates && templates->windowFlags().testFlag(Qt::Popup));
    QCOMPARE(window.findChild<QAction *>("openProjectAction"), open);
    QCOMPARE(window.findChild<QAction *>("saveProjectAction"), save);

    // InstantPopup's click runs a nested menu loop. Open nonblocking so QtTest
    // can drive the real tree row while the menu is visible.
    scopeButton->menu()->popup(scopeButton->mapToGlobal(QPoint(0, scopeButton->height())));
    QCoreApplication::processEvents();
    tree->setCurrentItem(outer->child(0));
    const QRect row = tree->visualItemRect(outer->child(0));
    QTest::mouseClick(tree->viewport(), Qt::LeftButton, Qt::NoModifier, row.center());
    QCoreApplication::processEvents();
    QCOMPARE(scene->scope(), QStringLiteral("inner"));
    QCOMPARE(scopeButton->text(), QStringLiteral("Subflow Proxy"));

    const auto reopenScope = [&] {
        bool visible = false;
        QTimer::singleShot(20, &window, [&] {
            visible = scopeButton->menu()->isVisible() && tree->isVisible() &&
                      tree->width() > 0 && tree->height() > 0;
            scopeButton->menu()->close();
        });
        QTest::mouseClick(scopeButton, Qt::LeftButton);
        QTest::qWait(25);
        QVERIFY(visible);
    };
    reopenScope();

    QVERIFY2(nn_app_rename_node(app, "outer", "Renamed flow", error, sizeof(error)), error);
    Q_EMIT scene->modelChanged();
    QCoreApplication::processEvents();
    root = tree->topLevelItem(0);
    QCOMPARE(root->child(0)->text(0), QStringLiteral("Renamed flow"));
    reopenScope();

    const NNModel *model = nn_app_model(app);
    std::vector<QByteArray> children;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node && node->scope_id && std::strcmp(node->scope_id, "inner") == 0)
            children.emplace_back(node->id);
    }
    for (const QByteArray &id : children)
        QVERIFY2(nn_app_remove_node(app, id.constData(), error, sizeof(error)), error);
    QVERIFY2(nn_app_remove_node(app, "inner", error, sizeof(error)), error);
    Q_EMIT scene->modelChanged();
    QCoreApplication::processEvents();
    QTreeWidgetItemIterator iterator(tree);
    while (*iterator) {
        QVERIFY((*iterator)->data(0, MainWindowUtils::IdRole).toString() != QStringLiteral("inner"));
        ++iterator;
    }
    QCOMPARE(scene->scope(), QString());
    reopenScope();
}

void WindowTest::graphHistoryTracksEditsAndScopeFallback() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "history-test", "History test", "0.1.0", false,
                           error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "source", "core.input", "0.1.0", "", 20, 40,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "layer", "core.relu", "0.1.0", "", 260, 300,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "edge", "source", "out", "layer", "in",
                            error, sizeof(error)), error);
    QVERIFY2(nn_app_save(app, error, sizeof(error)), error);
    const QByteArray projectPath = (temporary.path() + QStringLiteral("/history-test")).toUtf8();
    QVERIFY2(nn_app_open(app, projectPath.constData(), error, sizeof(error)), error);

    MainWindow window(app);
    window.show();
    QCoreApplication::processEvents();
    auto *scene = window.findChild<GraphScene *>();
    auto *undo = window.findChild<QAction *>("undoAction");
    auto *redo = window.findChild<QAction *>("redoAction");
    auto *arrange = window.findChild<QAction *>("arrange");
    auto *fit = window.findChild<QAction *>("fitGraph");
    QVERIFY(scene && undo && redo && arrange && fit);
    QVERIFY(!undo->isEnabled());
    QVERIFY(!redo->isEnabled());
    QCOMPARE(undo->shortcut(), QKeySequence::Undo);
    QCOMPARE(redo->shortcut(), QKeySequence::Redo);

    QVERIFY2(nn_app_move_node(app, "layer", 620, 380, error, sizeof(error)), error);
    Q_EMIT scene->modelChanged();
    QCoreApplication::processEvents();
    QVERIFY(undo->isEnabled());
    undo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(nn_model_find_node(nn_app_model(app), "layer")->x, 260);
    QCOMPARE(nn_model_find_node(nn_app_model(app), "layer")->y, 300);
    QVERIFY(!undo->isEnabled());
    QVERIFY(redo->isEnabled());
    redo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(nn_model_find_node(nn_app_model(app), "layer")->x, 620);
    QCOMPARE(nn_model_find_node(nn_app_model(app), "layer")->y, 380);
    QVERIFY2(nn_app_rename_node(app, "layer", "Renamed layer", error, sizeof(error)), error);
    Q_EMIT scene->modelChanged();
    QCoreApplication::processEvents();
    QCOMPARE(QString::fromUtf8(nn_model_find_node(nn_app_model(app), "layer")->label),
             QStringLiteral("Renamed layer"));
    undo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(QString::fromUtf8(nn_model_find_node(nn_app_model(app), "layer")->label),
             QStringLiteral("ReLU"));
    redo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(QString::fromUtf8(nn_model_find_node(nn_app_model(app), "layer")->label),
             QStringLiteral("Renamed layer"));
    undo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(QString::fromUtf8(nn_model_find_node(nn_app_model(app), "layer")->label),
             QStringLiteral("ReLU"));
    undo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(nn_model_find_node(nn_app_model(app), "layer")->x, 260);
    QCOMPARE(nn_model_find_node(nn_app_model(app), "layer")->y, 300);
    QVERIFY(!undo->isEnabled());
    QVERIFY(redo->isEnabled());

    std::map<QByteArray, QPoint> beforeArrange;
    const NNModel *model = nn_app_model(app);
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        beforeArrange.emplace(QByteArray(node->id), QPoint(node->x, node->y));
    }
    arrange->trigger();
    QCoreApplication::processEvents();
    QVERIFY(undo->isEnabled());
    const std::map<QByteArray, QPoint> afterArrange = [&] {
        std::map<QByteArray, QPoint> points;
        const NNModel *current = nn_app_model(app);
        for (size_t i = 0; i < nn_model_node_count(current); ++i) {
            const NNNode *node = nn_model_node_at(current, i);
            points.emplace(QByteArray(node->id), QPoint(node->x, node->y));
        }
        return points;
    }();
    fit->trigger();
    QCoreApplication::processEvents();
    QVERIFY(undo->isEnabled());
    for (const auto &entry : afterArrange) {
        const NNNode *node = nn_model_find_node(nn_app_model(app), entry.first.constData());
        QVERIFY(node);
        QCOMPARE(QPoint(node->x, node->y), entry.second);
    }
    undo->trigger();
    QCoreApplication::processEvents();
    QVERIFY(!undo->isEnabled());
    for (const auto &entry : beforeArrange) {
        const NNNode *node = nn_model_find_node(nn_app_model(app), entry.first.constData());
        QVERIFY(node);
        QCOMPARE(QPoint(node->x, node->y), entry.second);
    }
    redo->trigger();
    QCoreApplication::processEvents();
    QVERIFY(undo->isEnabled());
    for (const auto &entry : afterArrange) {
        const NNNode *node = nn_model_find_node(nn_app_model(app), entry.first.constData());
        QVERIFY(node);
        QCOMPARE(QPoint(node->x, node->y), entry.second);
    }

    QVERIFY2(nn_app_add_node(app, "flow", "core.subflow-proxy", "0.1.0", "", 0, 0,
                             error, sizeof(error)), error);
    Q_EMIT scene->modelChanged();
    QCoreApplication::processEvents();
    scene->setScope(QStringLiteral("flow"));
    QCOMPARE(scene->scope(), QStringLiteral("flow"));
    undo->trigger();
    QCoreApplication::processEvents();
    QCOMPARE(scene->scope(), QString());
    QVERIFY(!nn_model_find_node(nn_app_model(app), "flow"));
}

static void setInitialStereotypeLua(QDialog *form, const QString &source) {
    QTimer::singleShot(0, form, [form, source] {
        auto *editor = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        if (!editor || editor == form) return;
        auto *lua = editor->findChild<QPlainTextEdit *>("stereotypeLua");
        auto *apply = editor->findChild<QPushButton *>("applyInitialStereotypeSource");
        if (lua && apply) {
            lua->setPlainText(source);
            apply->click();
        }
    });
    auto *button = form->findChild<QPushButton *>("editStereotypeSourceButton");
    if (button) button->click();
}

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

struct DialogRenderEvidence {
    QColor background;
    QColor foreground;
    bool textRendered = false;
    bool captured = true;
};

static qreal contrastRatio(const QColor &foreground, const QColor &background) {
    auto luminance = [](const QColor &color) {
        auto channel = [](int value) {
            const qreal c = value / 255.0;
            return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
        };
        return 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) +
               0.0722 * channel(color.blue());
    };
    const qreal a = luminance(foreground), b = luminance(background);
    return (qMax(a, b) + 0.05) / (qMin(a, b) + 0.05);
}

static DialogRenderEvidence inspectDialog(QWidget *surface, QWidget *textWidget,
                                          const QString &capturePath = {}, const QRect &region = {}) {
    surface->ensurePolished();
    QApplication::processEvents();
    const QImage image = surface->grab().toImage().convertToFormat(QImage::Format_ARGB32);
    DialogRenderEvidence evidence;
    if (image.isNull()) return evidence;

    const QPoint textOrigin = textWidget->mapTo(surface, QPoint(0, 0));
    const QRect textRect = (region.isNull() ? QRect(textOrigin, textWidget->size())
                                             : region.translated(textOrigin)).intersected(image.rect());
    QMap<QRgb, int> cornerColors;
    for (int y = textRect.top(); y <= textRect.bottom(); ++y)
        for (int x = textRect.left(); x <= textRect.right(); ++x)
            ++cornerColors[image.pixel(x, y)];
    int mostCommon = 0;
    for (auto it = cornerColors.cbegin(); it != cornerColors.cend(); ++it) {
        if (it.value() > mostCommon) {
            mostCommon = it.value();
            evidence.background = QColor::fromRgba(it.key());
        }
    }

    evidence.foreground = textWidget->palette().color(QPalette::WindowText);
    for (int y = textRect.top(); y <= textRect.bottom() && !evidence.textRendered; ++y) {
        for (int x = textRect.left(); x <= textRect.right(); ++x) {
            const QColor pixel = image.pixelColor(x, y);
            if (qAbs(pixel.red() - evidence.foreground.red()) <= 12 &&
                qAbs(pixel.green() - evidence.foreground.green()) <= 12 &&
                qAbs(pixel.blue() - evidence.foreground.blue()) <= 12) {
                evidence.textRendered = true;
                break;
            }
        }
    }
    if (!capturePath.isEmpty()) evidence.captured = image.save(capturePath);
    return evidence;
}

void WindowTest::lifecycleAndInspector() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "qt-test", "Qt test", "0.1.0", true,
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
    auto *addNodeButton = window.findChild<QPushButton *>("addNodeButton");
    QVERIFY(addNodeButton);
    QVERIFY(!addNodeButton->isEnabled());
    QTreeWidgetItem *packageLeaf = nullptr;
    QTreeWidgetItem *packageGroup = nullptr;
    for (int group = 0; group < palette->topLevelItemCount(); ++group) {
        if (!packageGroup) packageGroup = palette->topLevelItem(group);
        if (palette->topLevelItem(group)->childCount() > 0) {
            packageLeaf = palette->topLevelItem(group)->child(0);
            break;
        }
    }
    QVERIFY(packageLeaf && packageGroup);
    palette->setCurrentItem(packageGroup);
    QCoreApplication::processEvents();
    QVERIFY(!addNodeButton->isEnabled());
    palette->setCurrentItem(packageLeaf);
    QCoreApplication::processEvents();
    QVERIFY(addNodeButton->isEnabled());
    palette->clearSelection();
    QCoreApplication::processEvents();
    QVERIFY(!addNodeButton->isEnabled());
    QVERIFY(!window.findChild<QWidget *>("resources"));
    auto *problemsToggle = window.findChild<QToolButton *>("problemsToggle");
    QVERIFY(problemsToggle);
    QVERIFY(!problemsToggle->isChecked());
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
    QVERIFY(!window.findChild<QWidget *>("resources"));
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i)
        QVERIFY(!diagnostics->topLevelItem(i)->text(0).contains(QStringLiteral("Success")));
    QVERIFY(diagnostics->topLevelItemCount() < 8);
    QVERIFY(scene->nodeItem("dense1"));

    // Failed project replacement must keep the active graph and scene.
    answerDialog(QMessageBox::Ok);
    QVERIFY(!window.openProject(temporary.path() + "/missing"));
    QCOMPARE(nn_model_node_count(nn_app_model(app)), size_t(10));
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
    QTreeWidgetItem *parameters = nullptr;
    for (int i = 0; i < inspector->topLevelItemCount(); ++i)
        if (inspector->topLevelItem(i)->text(0) == QStringLiteral("Parameters"))
            parameters = inspector->topLevelItem(i);
    QVERIFY(parameters);
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

void WindowTest::appDialogsContrastUnderDarkPalette() {
    struct PaletteRestore {
        QPalette palette = QApplication::palette();
        ~PaletteRestore() { QApplication::setPalette(palette); }
    } restore;

    QPalette dark = QApplication::palette();
    dark.setColor(QPalette::Window, QColor("#20242b"));
    dark.setColor(QPalette::WindowText, QColor("#f0f2f5"));
    dark.setColor(QPalette::Base, QColor("#15191f"));
    dark.setColor(QPalette::Text, QColor("#f0f2f5"));
    dark.setColor(QPalette::Button, QColor("#20242b"));
    dark.setColor(QPalette::ButtonText, QColor("#f0f2f5"));
    QApplication::setPalette(dark);

    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "dialog-theme", "Dialog theme", "0.1.0", false,
                           error, sizeof(error)), error);
    MainWindow window(app);
    window.show();

    const QString captureDirectory = qEnvironmentVariable("NN_TEST_DIALOG_CAPTURE_DIR");
    auto capturePath = [&captureDirectory](const QString &name) {
        return captureDirectory.isEmpty() ? QString() : QDir(captureDirectory).filePath(name + ".png");
    };
    auto checkEvidence = [](const DialogRenderEvidence &evidence, const char *widgetName) {
        QVERIFY(evidence.background.isValid());
        QCOMPARE(evidence.background, QColor("#eceff3"));
        QVERIFY(evidence.textRendered);
        QVERIFY(evidence.captured);
        QVERIFY2(contrastRatio(evidence.foreground, evidence.background) >= 4.5,
                 widgetName);
    };
    auto *problemsToggle = window.findChild<QToolButton *>("problemsToggle");
    QVERIFY(problemsToggle);
    QCOMPARE(problemsToggle->arrowType(), Qt::RightArrow);
    auto checkToggleRender = [&](const QString &state, QImage *buttonImage) {
        const auto evidence = inspectDialog(&window, problemsToggle, capturePath(QStringLiteral("problems-toggle-") + state));
        QVERIFY(evidence.background.isValid());
        QVERIFY(evidence.textRendered);
        QVERIFY(evidence.captured);
        QVERIFY2(contrastRatio(evidence.foreground, evidence.background) >= 4.5,
                 qPrintable(QStringLiteral("problems toggle %1 text must contrast with its rendered background").arg(state)));
        *buttonImage = window.grab().toImage().copy(
            QRect(problemsToggle->mapTo(&window, QPoint(0, 0)), problemsToggle->size()));
    };
    QImage collapsedButton;
    checkToggleRender(QStringLiteral("normal"), &collapsedButton);
    QTest::mouseMove(problemsToggle, problemsToggle->rect().center());
    QCoreApplication::processEvents();
    QImage hoverButton;
    checkToggleRender(QStringLiteral("hover"), &hoverButton);
    QVERIFY(!hoverButton.isNull());
    QTest::mousePress(problemsToggle, Qt::LeftButton, Qt::NoModifier, problemsToggle->rect().center());
    QCoreApplication::processEvents();
    QImage pressedButton;
    checkToggleRender(QStringLiteral("pressed"), &pressedButton);
    QVERIFY(!pressedButton.isNull());
    QTest::mouseRelease(problemsToggle, Qt::LeftButton, Qt::NoModifier, problemsToggle->rect().center());
    QTRY_VERIFY(problemsToggle->isChecked());
    QCOMPARE(problemsToggle->arrowType(), Qt::DownArrow);
    QImage expandedButton;
    checkToggleRender(QStringLiteral("expanded"), &expandedButton);
    QVERIFY(collapsedButton != expandedButton);
    problemsToggle->click();
    QTRY_VERIFY(!problemsToggle->isChecked());
    QCOMPARE(problemsToggle->arrowType(), Qt::RightArrow);

    auto openAuthoringDialog = [&](const QString &actionText, const char *checkBoxName,
                                   const QString &captureName) {
        DialogRenderEvidence labelEvidence, checkBoxEvidence;
        DialogRenderEvidence selectedTab, otherTab, selectedPython;
        QTimer::singleShot(0, &window, [&] {
            auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            if (!dialog) return;
            QLabel *label = nullptr;
            for (QLabel *candidate : dialog->findChildren<QLabel *>())
                if (!candidate->text().isEmpty()) { label = candidate; break; }
            if (label)
                labelEvidence = inspectDialog(dialog, label, capturePath(captureName));
            if (checkBoxName) {
                if (auto *checkBox = dialog->findChild<QCheckBox *>(checkBoxName))
                    checkBoxEvidence = inspectDialog(dialog, checkBox);
            }
            if (captureName == QStringLiteral("stereotype")) {
                QTimer::singleShot(0, dialog, [&, dialog] {
                    auto *editor = qobject_cast<QDialog *>(QApplication::activeModalWidget());
                    auto *tabs = editor ? editor->findChild<QTabWidget *>() : nullptr;
                    if (!editor || !tabs) return;
                    auto *bar = tabs->tabBar();
                    selectedTab = inspectDialog(editor, bar, {}, bar->tabRect(0));
                    otherTab = inspectDialog(editor, bar, {}, bar->tabRect(1));
                    tabs->setCurrentIndex(1);
                    selectedPython = inspectDialog(editor, bar, {}, bar->tabRect(1));
                    editor->reject();
                });
                if (auto *sourceButton = dialog->findChild<QPushButton *>("editStereotypeSourceButton"))
                    sourceButton->click();
            }
            dialog->reject();
        });
        QAction *action = nullptr;
        for (QAction *candidate : window.findChildren<QAction *>())
            if (candidate->text() == actionText) action = candidate;
        QVERIFY(action);
        action->trigger();
        checkEvidence(labelEvidence, "dialog label text needs 4.5:1 contrast against the rendered surface");
        if (checkBoxName)
            checkEvidence(checkBoxEvidence, "dialog checkbox text needs 4.5:1 contrast against the rendered surface");
        if (captureName == QStringLiteral("stereotype")) {
            for (const auto &tab : {selectedTab, otherTab, selectedPython}) {
                QVERIFY(tab.background.isValid());
                QVERIFY(tab.textRendered);
                QVERIFY2(contrastRatio(tab.foreground, tab.background) >= 4.5,
                         "tab text needs 4.5:1 contrast against its rendered surface");
            }
        }
    };

    openAuthoringDialog(QStringLiteral("Create stereotype…"), "stereotypeOutputOverride", QStringLiteral("stereotype"));
    openAuthoringDialog(QStringLiteral("Create dataset…"), "datasetSelect", QStringLiteral("dataset"));

    DialogRenderEvidence chooserEvidence;
    QTimer::singleShot(0, &window, [&] {
        auto *chooser = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
        if (!chooser) return;
        QLabel *label = nullptr;
        for (QLabel *candidate : chooser->findChildren<QLabel *>())
            if (!candidate->text().isEmpty()) { label = candidate; break; }
        if (label)
            chooserEvidence = inspectDialog(chooser, label, capturePath(QStringLiteral("project-chooser")));
        chooser->button(QMessageBox::Close)->click();
    });
    window.showProjectChooser();
    checkEvidence(chooserEvidence, "project chooser text needs 4.5:1 contrast against the rendered surface");
    window.close();
}

void WindowTest::modelProblemDiagnosticsAndNavigation() {
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray example = QByteArray(NN_SOURCE_DIR) + "/examples/models/mnist-mlp";
    QVERIFY2(nn_app_open(app, example.constData(), error, sizeof(error)), error);
    QVERIFY2(nn_app_set_parameter_text(app, "dense1", "in_features", "785",
                                       error, sizeof(error)), error);
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    QVERIFY2(report, error);
    const NNInferenceResult *rootResult = nullptr;
    const NNInferenceResult *blockedResult = nullptr;
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (!result || !result->node_id) continue;
        if (std::strcmp(result->node_id, "dense1") == 0) rootResult = result;
        if (std::strcmp(result->node_id, "output") == 0) blockedResult = result;
    }
    QVERIFY(rootResult && rootResult->status == NN_INFERENCE_SEMANTIC_ERROR);
    QVERIFY(blockedResult && blockedResult->status == NN_INFERENCE_UNRESOLVED);
    QVERIFY(blockedResult->cause_node_id && std::strcmp(blockedResult->cause_node_id, "dense1") == 0);
    QVERIFY2(nn_app_add_node(app, "flow", "core.subflow-proxy", "0.1.0", "", 20, 20,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "flow-child", "core.relu", "0.1.0", "flow", 20, 20,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_remove_node(app, "loss-output", error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "late-input", "core.input", "0.1.0", "", 20, 360,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_set_parameter_text(app, "late-input", "binding", "missing",
                                       error, sizeof(error)), error);

    MainWindow window(app);
    auto *scene = window.findChild<GraphScene *>();
    auto *diagnostics = window.findChild<QTreeWidget *>("diagnostics");
    auto *filter = window.findChild<QCheckBox *>("currentScopeProblems");
    auto *toggle = window.findChild<QToolButton *>("problemsToggle");
    QVERIFY(scene && diagnostics && filter && toggle);
    QCOMPARE(diagnostics->columnCount(), 4);
    QCOMPARE(diagnostics->headerItem()->text(0), QStringLiteral("Category"));
    QCOMPARE(diagnostics->headerItem()->text(1), QStringLiteral("Node"));
    QCOMPARE(diagnostics->headerItem()->text(2), QStringLiteral("Scope"));
    QCOMPARE(diagnostics->headerItem()->text(3), QStringLiteral("Reason"));
    QVERIFY(!toggle->isChecked());
    window.show();
    QCoreApplication::processEvents();
    QVERIFY(!diagnostics->isVisibleTo(&window));
    toggle->click();
    QCoreApplication::processEvents();
    QVERIFY(toggle->isChecked());
    QVERIFY(diagnostics->isVisibleTo(&window));
    const QString globalCounts = toggle->accessibleName();
    QVERIFY(globalCounts.contains(QStringLiteral("Errors:")));
    QVERIFY(globalCounts.contains(QStringLiteral("warnings:")));
    QVERIFY(globalCounts.contains(QStringLiteral("internal failures:")));
    QVERIFY(globalCounts.section(QStringLiteral("Errors: "), 1).section(QLatin1Char(','), 0, 0).toInt() > 0);
    QVERIFY(globalCounts.section(QStringLiteral("warnings: "), 1).section(QLatin1Char(','), 0, 0).toInt() > 0);
    QCOMPARE(globalCounts.section(QStringLiteral("internal failures: "), 1).toInt(), 0);
    QVERIFY(diagnostics->topLevelItemCount() >= 2);
    int lateInputIndex = -1;
    int denseIndex = -1;
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i) {
        const QString id = diagnostics->topLevelItem(i)->data(0, Qt::UserRole).toString();
        if (id == QStringLiteral("late-input")) lateInputIndex = i;
        if (id == QStringLiteral("dense1")) denseIndex = i;
    }
    QVERIFY(lateInputIndex >= 0 && denseIndex >= 0);
    QVERIFY2(lateInputIndex < denseIndex,
             "A newly appended source problem must precede older downstream errors.");
    int countedErrors = 0;
    int countedWarnings = 0;
    int countedInternal = 0;
    for (QTreeWidgetItemIterator it(diagnostics); *it; ++it) {
        const QString category = (*it)->text(0);
        if (category.contains(QStringLiteral("Model error")) ||
            category.contains(QStringLiteral("Lua compilation error"))) ++countedErrors;
        else if (category.contains(QStringLiteral("Incomplete"))) ++countedWarnings;
        else if (category.contains(QStringLiteral("Analysis unavailable"))) ++countedInternal;
    }
    QCOMPARE(globalCounts.section(QStringLiteral("Errors: "), 1).section(QLatin1Char(','), 0, 0).toInt(), countedErrors);
    QCOMPARE(globalCounts.section(QStringLiteral("warnings: "), 1).section(QLatin1Char(','), 0, 0).toInt(), countedWarnings);
    QCOMPARE(globalCounts.section(QStringLiteral("internal failures: "), 1).toInt(), countedInternal);
    QVERIFY(diagnostics->topLevelItem(diagnostics->topLevelItemCount() - 1)
                ->data(0, Qt::UserRole).toString().isEmpty());
    QVERIFY(diagnostics->topLevelItem(diagnostics->topLevelItemCount() - 1)
                ->text(1) == QStringLiteral("Root"));
    QTreeWidgetItem *root = nullptr;
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i)
        if (diagnostics->topLevelItem(i)->data(0, Qt::UserRole).toString() == QStringLiteral("dense1"))
            root = diagnostics->topLevelItem(i);
    QVERIFY(root);
    QCOMPARE(root->data(0, Qt::UserRole).toString(), QStringLiteral("dense1"));
    QVERIFY(root->childCount() > 0);
    QVERIFY(!root->isExpanded());
    QCOMPARE(scene->nodeItem("dense1")->problemCategory(), QStringLiteral("model"));
    QCOMPARE(scene->nodeItem("output")->problemCategory(), QStringLiteral("incomplete"));
    QVERIFY(diagnostics->wordWrap());
    const NNNode *denseNode = nn_model_find_node(nn_app_model(app), "dense1");
    QVERIFY(denseNode && denseNode->label);
    const QString denseLabel = QString::fromUtf8(denseNode->label);
    QString columnMetrics;
    auto columnsFit = [diagnostics, &columnMetrics](int minReasonWidth) {
        auto *header = diagnostics->header();
        const int viewportWidth = diagnostics->viewport()->width();
        const QStringList names = {QStringLiteral("Category"), QStringLiteral("Node"),
                                   QStringLiteral("Scope"), QStringLiteral("Reason")};
        QStringList sections;
        int previousEnd = 0;
        bool allVisible = true;
        for (int column = 0; column < diagnostics->columnCount(); ++column) {
            const int start = header->sectionViewportPosition(column);
            const int width = header->sectionSize(column);
            const int end = start + width;
            sections.push_back(QStringLiteral("%1=%2px@%3").arg(names.value(column)).arg(width).arg(start));
            if (start < previousEnd || end > viewportWidth) allVisible = false;
            previousEnd = end;
        }
        const int categoryWidth = header->sectionSize(0);
        const int reasonWidth = header->sectionSize(3);
        columnMetrics = QStringLiteral("viewport=%1px, %2")
            .arg(viewportWidth).arg(sections.join(QStringLiteral(", ")));
        return viewportWidth > 0 && categoryWidth > 0 && reasonWidth >= minReasonWidth &&
            categoryWidth * 2 < viewportWidth && allVisible;
    };
    window.resize(1280, 800);
    QCoreApplication::processEvents();
    const bool desktopColumnsFit = columnsFit(180);
    QVERIFY2(desktopColumnsFit, qPrintable(QStringLiteral("Columns should fit at 1280x800: %1").arg(columnMetrics)));
    const QString normalCapture = qEnvironmentVariable("NN_PROBLEMS_CAPTURE");
    if (!normalCapture.isEmpty()) QVERIFY(window.grab().save(normalCapture));
    window.resize(900, 560);
    QCoreApplication::processEvents();
    const bool smallColumnsFit = columnsFit(120);
    QVERIFY2(smallColumnsFit, qPrintable(QStringLiteral("Columns should fit at 900x560: %1").arg(columnMetrics)));
    const QString smallCapture = qEnvironmentVariable("NN_PROBLEMS_CAPTURE_SMALL");
    if (!smallCapture.isEmpty()) QVERIFY(window.grab().save(smallCapture));
    const QString rowCategory = root->text(0);
    QVERIFY(rowCategory.contains(QStringLiteral("Model error")));
    QCOMPARE(root->text(1), denseLabel);
    QCOMPARE(root->text(2), QStringLiteral("Root"));
    QVERIFY(!root->text(3).trimmed().isEmpty());
    QCOMPARE(root->toolTip(3), root->text(3));
    const QRect rootRect = diagnostics->visualItemRect(root);
    QVERIFY(rootRect.isValid());
    QVERIFY(rootRect.height() >= diagnostics->fontMetrics().lineSpacing());

    scene->setScope(QStringLiteral("flow"));
    filter->setChecked(true);
    QCoreApplication::processEvents();
    QVERIFY(diagnostics->topLevelItemCount() > 0);
    QCOMPARE(toggle->accessibleName(), globalCounts);
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i) {
        const QString id = diagnostics->topLevelItem(i)->data(0, Qt::UserRole).toString();
        QVERIFY(!id.isEmpty());
        QVERIFY(id != QStringLiteral("dense1"));
    }

    filter->setChecked(false);
    // Problem rows share the automation reveal path and preserve copied stable IDs.
    root = nullptr;
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i)
        if (diagnostics->topLevelItem(i)->data(0, Qt::UserRole).toString() == QStringLiteral("dense1"))
            root = diagnostics->topLevelItem(i);
    QVERIFY(root);
    root->setExpanded(true);
    diagnostics->itemClicked(root, 0);
    QTRY_COMPARE(scene->scope(), QString());
    QTRY_COMPARE(scene->selectedNodeId(), QStringLiteral("dense1"));
    toggle->click();
    QCoreApplication::processEvents();
    QVERIFY(!toggle->isChecked());
    QVERIFY(!diagnostics->isVisibleTo(&window));
}

void WindowTest::toolbarDeleteTracksSelection() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "delete-toolbar", "Delete toolbar", "0.1.0",
                           false, error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "layer", "core.relu", "0.1.0", "", 100, 100,
                             error, sizeof(error)), error);

    MainWindow window(app);
    auto *scene = window.findChild<GraphScene *>();
    auto *action = window.findChild<QAction *>("deleteSelection");
    auto *button = window.findChild<QToolButton *>("deleteSelectionButton");
    QVERIFY(scene && action && button);
    QVERIFY(!action->isEnabled());
    scene->nodeItem(QStringLiteral("layer"))->setSelected(true);
    QCoreApplication::processEvents();
    QVERIFY(action->isEnabled());
    QVERIFY(button->styleSheet().contains(QStringLiteral("#b23b35")));
    action->trigger();
    QTRY_VERIFY(!nn_model_find_node(nn_app_model(app), "layer"));
}

void WindowTest::retainedLuaValidationFormAndNonblockingRejection() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "lua-retained", "Lua retained", "0.1.0", false,
                           error, sizeof(error)), error);
    MainWindow window(app);
    QAction *action = nullptr;
    for (QAction *candidate : window.findChildren<QAction *>())
        if (candidate->text() == QString::fromUtf8("Create stereotype…")) action = candidate;
    QVERIFY(action);
    QTimer::singleShot(0, &window, [&window] {
        auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(dialog);
        dialog->findChild<QLineEdit *>("stereotypeId")->setText("bad.lua");
        dialog->findChild<QLineEdit *>("stereotypeVersion")->setText("1.0.0");
        dialog->findChild<QLineEdit *>("stereotypeName")->setText("Malformed Lua");
        setInitialStereotypeLua(dialog, QStringLiteral("return function("));
        QTimer::singleShot(0, dialog, [dialog] {
            auto *buttons = dialog->findChild<QDialogButtonBox *>();
            buttons->button(QDialogButtonBox::Save)->click();
            QTimer::singleShot(0, dialog, [dialog] {
                QVERIFY(dialog->isVisible());
                QCOMPARE(dialog->findChild<QLineEdit *>("stereotypeId")->text(), QStringLiteral("bad.lua"));
                auto *luaError = dialog->findChild<QLabel *>("stereotypeLuaError");
                QVERIFY(!luaError->text().isEmpty());
                dialog->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Cancel)->click();
            });
        });
    });
    action->trigger();
    QVERIFY(!nn_catalog_find(nn_project_catalog(nn_app_project(app)), "bad.lua", "1.0.0"));

    QAction *datasetAction = nullptr;
    for (QAction *candidate : window.findChildren<QAction *>())
        if (candidate->text() == QString::fromUtf8("Create dataset…")) datasetAction = candidate;
    QVERIFY(datasetAction);
    QTimer::singleShot(0, &window, [&window] {
        auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(dialog);
        dialog->findChild<QLineEdit *>("datasetId")->setText("retained.dataset");
        dialog->findChild<QLineEdit *>("datasetVersion")->setText("1.0.0");
        QTimer::singleShot(0, dialog, [dialog] {
            dialog->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save)->click();
            QTimer::singleShot(0, dialog, [dialog] {
                QVERIFY(dialog->isVisible());
                QCOMPARE(dialog->findChild<QLineEdit *>("datasetId")->text(),
                         QStringLiteral("retained.dataset"));
                QVERIFY(!dialog->findChild<QLabel *>("datasetError")->text().isEmpty());
                dialog->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Cancel)->click();
            });
        });
    });
    datasetAction->trigger();
    QVERIFY(!nn_project_dataset_count(nn_app_project(app)));

    auto *scene = window.findChild<GraphScene *>();
    QVERIFY(scene);
    scene->errorOccurred(QStringLiteral("Cannot connect nodes: Target input is occupied."));
    QCoreApplication::processEvents();
    QMessageBox *rejection = nullptr;
    for (QMessageBox *candidate : window.findChildren<QMessageBox *>())
        if (candidate->isVisible() && candidate->text().startsWith(QStringLiteral("Cannot connect nodes:")))
            rejection = candidate;
    QVERIFY(rejection);
    QVERIFY(!rejection->isModal());
    rejection->close();
    window.close();
}

void WindowTest::currentScopeRetainsOutsideCauseContext() {
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray project = QByteArray(NN_SOURCE_DIR) + "/examples/models/mnist-vae";
    QVERIFY2(nn_app_open(app, project.constData(), error, sizeof(error)), error);
    QVERIFY2(nn_app_set_parameter_text(app, "mean", "out_features", "31",
                                       error, sizeof(error)), error);
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    QVERIFY2(report, error);

    QString targetId, targetScope, causeId, causeScope;
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (!result || !result->node_id || !result->cause_node_id || !*result->cause_node_id) continue;
        const NNNode *target = nn_model_find_node(nn_app_model(app), result->node_id);
        const NNNode *cause = nn_model_find_node(nn_app_model(app), result->cause_node_id);
        if (!target || !cause) continue;
        const QString targetScopeId = QString::fromUtf8(target->scope_id ? target->scope_id : "");
        const QString causeScopeId = QString::fromUtf8(cause->scope_id ? cause->scope_id : "");
        if (targetScopeId == causeScopeId) continue;
        const NNInferenceResult *root = nullptr;
        for (size_t j = 0; j < nn_inference_count(report); ++j) {
            const NNInferenceResult *candidate = nn_inference_at(report, j);
            if (candidate && candidate->node_id &&
                std::strcmp(candidate->node_id, result->cause_node_id) == 0) root = candidate;
        }
        if (!root || root->cause_node_id ||
            std::strcmp(nn_inference_category(root->status), "model") != 0) continue;
        targetId = QString::fromUtf8(result->node_id);
        targetScope = targetScopeId;
        causeId = QString::fromUtf8(result->cause_node_id);
        causeScope = causeScopeId;
        break;
    }
    QVERIFY(!targetId.isEmpty());
    QVERIFY(targetScope != causeScope);

    MainWindow window(app);
    auto *scene = window.findChild<GraphScene *>();
    auto *diagnostics = window.findChild<QTreeWidget *>("diagnostics");
    auto *filter = window.findChild<QCheckBox *>("currentScopeProblems");
    QVERIFY(scene && diagnostics && filter);
    scene->setScope(targetScope);
    filter->setChecked(true);
    QCoreApplication::processEvents();

    QTreeWidgetItem *causeItem = nullptr;
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i)
        if (diagnostics->topLevelItem(i)->data(0, Qt::UserRole).toString() == causeId)
            causeItem = diagnostics->topLevelItem(i);
    QVERIFY(causeItem);
    QVERIFY(causeItem->text(1).contains(QStringLiteral("cause context"), Qt::CaseInsensitive));
    QVERIFY(causeItem->text(0).contains(QStringLiteral("Model error")));
    bool hasCurrentBlockedNode = false;
    for (int i = 0; i < causeItem->childCount(); ++i)
        if (causeItem->child(i)->data(0, Qt::UserRole).toString() == targetId)
            hasCurrentBlockedNode = true;
    QVERIFY(hasCurrentBlockedNode);

    scene->setScope(causeScope);
    QCoreApplication::processEvents();
    NodeItem *causeNode = scene->nodeItem(causeId);
    QVERIFY(causeNode);
    QCOMPARE(causeNode->problemCategory(), QStringLiteral("model"));
}

void WindowTest::visualResourceAuthoringForms() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "author-test", "Authoring test", "0.1.0", false,
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
        auto *sourceButton = dialog->findChild<QPushButton *>("editStereotypeSourceButton");
        QVERIFY(sourceButton);
        setInitialStereotypeLua(dialog, QStringLiteral(
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
    QCOMPARE(nn_model_node_count(nn_app_model(app)), size_t(3));
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

void WindowTest::datasetManagerEditFormAndRetainedValidation() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "dataset-manager", "Dataset manager", "0.1.0",
                           false, error, sizeof(error)), error);
    QVERIFY2(nn_app_create_dataset(app, "manager.data", "1.0.0",
        R"({"name":"Manager data","description":"Original","batch":{"inputs":{"features":{"dtype":"float32","shape":["B",4]}},"targets":{}}})",
        true, error, sizeof(error)), error);
    MainWindow window(app);
    QAction *manage = nullptr;
    for (QAction *candidate : window.findChildren<QAction *>())
        if (candidate->text() == QStringLiteral("Manage datasets…")) manage = candidate;
    QVERIFY(manage);
    QTimer::singleShot(0, &window, [&window, &app] {
        auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(manager);
        QCOMPARE(manager->objectName(), QStringLiteral("datasetManager"));
        auto *list = manager->findChild<QListWidget *>("datasetManagerList");
        QVERIFY(list && list->count() == 1);
        auto *edit = manager->findChild<QPushButton *>("datasetManagerEdit");
        auto *select = manager->findChild<QPushButton *>("datasetManagerActivate");
        auto *create = manager->findChild<QPushButton *>("datasetManagerNew");
        QVERIFY(edit && select && create);
        list->clearSelection();
        list->setCurrentItem(nullptr);
        QVERIFY(!edit->isEnabled() && !select->isEnabled());
        QTimer::singleShot(0, manager, [manager] {
            auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            QVERIFY(form);
            form->findChild<QLineEdit *>("datasetId")->setText("created.via.manager");
            form->findChild<QLineEdit *>("datasetName")->setText("Created by manager");
            form->findChild<QCheckBox *>("datasetSelect")->setChecked(false);
            auto *inputs = form->findChild<QTableWidget *>("datasetInputs");
            QVERIFY(inputs);
            QPushButton *addRow = nullptr;
            for (QPushButton *button : form->findChildren<QPushButton *>())
                if (button->text() == QStringLiteral("Add row")) { addRow = button; break; }
            QVERIFY(addRow);
            addRow->click();
            inputs->item(0, 0)->setText("data");
            inputs->item(0, 2)->setText("B, 3");
            form->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save)->click();
        });
        create->click();
        QCOMPARE(list->count(), 2);
        QListWidgetItem *newDataset = nullptr;
        for (int i = 0; i < list->count(); ++i)
            if (list->item(i)->data(Qt::UserRole).toString() == QStringLiteral("created.via.manager"))
                newDataset = list->item(i);
        QVERIFY(newDataset);
        list->setCurrentItem(newDataset);
        QCOMPARE(QString::fromUtf8(nn_project_active_dataset(nn_app_project(app))->id),
                 QStringLiteral("manager.data"));
        QVERIFY(select->isEnabled());
        select->click();
        QVERIFY(select->isEnabled());
        select->click(); // Re-selecting the active row still reaches C synchronization.
        QVERIFY(select->isEnabled());

        QListWidgetItem *existing = nullptr;
        for (int i = 0; i < list->count(); ++i)
            if (list->item(i)->data(Qt::UserRole).toString() == QStringLiteral("manager.data"))
                existing = list->item(i);
        QVERIFY(existing);
        list->setCurrentItem(existing);
        QVERIFY(edit->isEnabled());
        QTimer::singleShot(0, manager, [manager] {
            auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            QVERIFY(form);
            QCOMPARE(form->findChild<QLineEdit *>("datasetId")->text(), QStringLiteral("manager.data"));
            QCOMPARE(form->findChild<QLineEdit *>("datasetVersion")->text(), QStringLiteral("1.0.0"));
            QVERIFY(form->findChild<QLineEdit *>("datasetId")->isReadOnly());
            QVERIFY(form->findChild<QLineEdit *>("datasetVersion")->isReadOnly());
            QCOMPARE(form->findChild<QLineEdit *>("datasetName")->text(), QStringLiteral("Manager data"));
            form->findChild<QLineEdit *>("datasetName")->setText("Edited in manager");
            form->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save)->click();
        });
        edit->click();
        manager->reject();
    });
    manage->trigger();
    QTimer::singleShot(0, &window, [&window, &app] {
        auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(manager);
        auto *list = manager->findChild<QListWidget *>("datasetManagerList");
        auto *activate = manager->findChild<QPushButton *>("datasetManagerActivate");
        QVERIFY(list && activate);
        QListWidgetItem *managerDataset = nullptr;
        for (int i = 0; i < list->count(); ++i)
            if (list->item(i)->data(Qt::UserRole).toString() == QStringLiteral("manager.data"))
                managerDataset = list->item(i);
        QVERIFY(managerDataset);
        list->setCurrentItem(managerDataset);
        QVERIFY(activate->isEnabled());
        activate->click();
        QCOMPARE(QString::fromUtf8(nn_project_active_dataset(nn_app_project(app))->id),
                 QStringLiteral("manager.data"));
        manager->reject();
    });
    manage->trigger();
    char *definition = nn_app_dataset_definition(app, "manager.data", "1.0.0", error, sizeof(error));
    QVERIFY(definition);
    QCOMPARE(QJsonDocument::fromJson(definition).object().value("name").toString(),
             QStringLiteral("Edited in manager"));
    nn_app_free_text(definition);

    QTimer::singleShot(0, &window, [&window] {
        auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(manager);
        auto *list = manager->findChild<QListWidget *>("datasetManagerList");
        QVERIFY(list && list->count() == 2);
        for (int i = 0; i < list->count(); ++i)
            if (list->item(i)->data(Qt::UserRole).toString() == QStringLiteral("manager.data"))
                list->setCurrentItem(list->item(i));
        auto *edit = manager->findChild<QPushButton *>("datasetManagerEdit");
        QVERIFY(edit);
        QTimer::singleShot(0, manager, [manager] {
            auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            QVERIFY(form);
            form->findChild<QLineEdit *>("datasetName")->setText("Must be cancelled");
            auto *inputs = form->findChild<QTableWidget *>("datasetInputs");
            QVERIFY(inputs);
            while (inputs->rowCount()) inputs->removeRow(0);
            QTimer::singleShot(0, form, [form] {
                form->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save)->click();
                QTimer::singleShot(0, form, [form] {
                    QVERIFY(form->isVisible());
                    QVERIFY(!form->findChild<QLabel *>("datasetError")->text().isEmpty());
                    QCOMPARE(form->findChild<QLineEdit *>("datasetName")->text(),
                             QStringLiteral("Must be cancelled"));
                    form->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Cancel)->click();
                });
            });
        });
        edit->click();
        manager->reject();
    });
    manage->trigger();
    definition = nn_app_dataset_definition(app, "manager.data", "1.0.0", error, sizeof(error));
    QVERIFY(definition);
    QCOMPARE(QJsonDocument::fromJson(definition).object().value("name").toString(),
             QStringLiteral("Edited in manager"));
    nn_app_free_text(definition);
    QVERIFY(nn_project_active_dataset(nn_app_project(app)));
    QCOMPARE(QString::fromUtf8(nn_project_active_dataset(nn_app_project(app))->id),
             QStringLiteral("manager.data"));
    QTimer::singleShot(0, &window, [&window, &app] {
        auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        auto *list = manager->findChild<QListWidget *>("datasetManagerList");
        auto *remove = manager->findChild<QPushButton *>("datasetManagerDelete");
        QVERIFY(list && remove && !remove->isEnabled());
        list->setCurrentRow(0);
        QVERIFY(remove->isEnabled());
        QTimer::singleShot(0, manager, [] {
            auto *confirmation = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
            QVERIFY(confirmation);
            confirmation->button(QMessageBox::Cancel)->click();
        });
        remove->click();
        QCOMPARE(list->count(), 2);
        QTimer::singleShot(0, manager, [] {
            auto *confirmation = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
            QVERIFY(confirmation);
            confirmation->button(QMessageBox::Yes)->click();
        });
        remove->click();
        QCOMPARE(list->count(), 1);
        QVERIFY(!nn_project_active_dataset(nn_app_project(app)));
        manager->reject();
    });
    manage->trigger();
}

void WindowTest::datasetImportFromEmptyDirectory() {
    struct NativeDialogSetting {
        bool previous = QCoreApplication::testAttribute(Qt::AA_DontUseNativeDialogs);
        NativeDialogSetting() { QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, true); }
        ~NativeDialogSetting() { QCoreApplication::setAttribute(Qt::AA_DontUseNativeDialogs, previous); }
    } nativeDialogSetting;
    QTemporaryDir projectParent;
    QTemporaryDir source;
    QVERIFY(projectParent.isValid() && source.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = projectParent.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "empty-import", "Empty import", "0.1.0",
                           false, error, sizeof(error)), error);
    MainWindow window(app);
    QAction *manage = nullptr;
    for (QAction *candidate : window.findChildren<QAction *>())
        if (candidate->text() == QStringLiteral("Manage datasets…")) manage = candidate;
    QVERIFY(manage);

    QTimer::singleShot(0, &window, [&window, &source] {
        auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(manager);
        auto *importButton = manager->findChild<QPushButton *>("datasetManagerImport");
        QVERIFY(importButton);
        QTimer::singleShot(0, manager, [manager, &source] {
            auto *picker = qobject_cast<QFileDialog *>(QApplication::activeModalWidget());
            QVERIFY(picker);
            picker->setDirectory(source.path());
            picker->selectFile(source.path());
            QVERIFY(QMetaObject::invokeMethod(picker, "accept", Qt::DirectConnection));
            QTimer::singleShot(0, manager, [manager] {
                auto *form = qobject_cast<QDialog *>(QApplication::activeModalWidget());
                QVERIFY(form);
                QCOMPARE(form->objectName(), QStringLiteral("datasetImportDialog"));
                QVERIFY(form->findChild<QLabel *>("datasetImportNotice")->text().contains(
                    QStringLiteral("nonfunctional scaffold")));
                auto *inputs = form->findChild<QTableWidget *>("datasetImportInputs");
                QVERIFY(inputs && inputs->rowCount() == 0);
                form->findChild<QLineEdit *>("datasetImportId")->setText("empty.dataset");
                form->findChild<QLineEdit *>("datasetImportName")->setText("Empty dataset");
                QPushButton *addRow = nullptr;
                for (QPushButton *button : form->findChildren<QPushButton *>())
                    if (button->text() == QStringLiteral("Add row")) { addRow = button; break; }
                QVERIFY(addRow);
                addRow->click();
                inputs->item(0, 0)->setText("features");
                inputs->item(0, 2)->setText("B, 4");
                form->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save)->click();
            });
        });
        importButton->click();
        QCOMPARE(manager->findChild<QListWidget *>("datasetManagerList")->count(), 1);
        manager->reject();
    });
    manage->trigger();
    QCOMPARE(nn_project_dataset_count(nn_app_project(app)), size_t(1));
    QVERIFY(nn_project_active_dataset(nn_app_project(app)));
    QCOMPARE(QString::fromUtf8(nn_project_active_dataset(nn_app_project(app))->id),
             QStringLiteral("empty.dataset"));
}

void WindowTest::automaticDatasetBindingInspectorAndActivation() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "auto-binding", "Automatic binding", "0.1.0",
                           false, error, sizeof(error)), error);
    const QByteArray first = R"({"name":"First","batch":{"inputs":{"features":{"dtype":"float32","shape":["B",4]}},"targets":{}}})";
    const QByteArray second = R"({"name":"Second","batch":{"inputs":{"pixels":{"dtype":"float32","shape":["B",784]}},"targets":{}}})";
    const QByteArray multiple = R"({"name":"Multiple","batch":{"inputs":{"features":{"dtype":"float32","shape":["B",4]},"target":{"dtype":"float32","shape":["B",1]}},"targets":{}}})";
    QVERIFY2(nn_app_create_dataset(app, "auto.first", "1.0.0", first.constData(), true,
                                   error, sizeof(error)), error);
    QVERIFY2(nn_app_create_dataset(app, "auto.second", "1.0.0", second.constData(), false,
                                   error, sizeof(error)), error);
    QVERIFY2(nn_app_create_dataset(app, "auto.multiple", "1.0.0", multiple.constData(), false,
                                   error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "single-input", "core.input", "0.1.0", "", 0, 0,
                             error, sizeof(error)), error);
    auto binding = [app] {
        char *value = nn_app_parameter_text(app, "single-input", "binding");
        const QString result = value ? QString::fromUtf8(value) : QString();
        nn_app_free_text(value);
        return result;
    };
    QCOMPARE(binding(), QStringLiteral("features"));

    MainWindow window(app);
    window.show();
    auto *scene = window.findChild<GraphScene *>();
    auto *inspector = window.findChild<QTreeWidget *>("inspector");
    QAction *manage = nullptr;
    for (QAction *candidate : window.findChildren<QAction *>())
        if (candidate->text() == QStringLiteral("Manage datasets…")) manage = candidate;
    QVERIFY(scene && inspector && manage);
    auto bindingEditor = [inspector]() -> QLineEdit * {
        for (int i = 0; i < inspector->topLevelItemCount(); ++i) {
            QTreeWidgetItem *parameters = inspector->topLevelItem(i);
            if (parameters->text(0) != QStringLiteral("Parameters")) continue;
            for (int j = 0; j < parameters->childCount(); ++j) {
                QTreeWidgetItem *row = parameters->child(j);
                if (row->text(0) == QStringLiteral("binding"))
                    return qobject_cast<QLineEdit *>(inspector->itemWidget(row, 1));
            }
        }
        return nullptr;
    };
    scene->nodeItem(QStringLiteral("single-input"))->setSelected(true);
    QCoreApplication::processEvents();
    QLineEdit *editor = bindingEditor();
    QVERIFY(editor);
    QVERIFY(editor->isReadOnly());
    QCOMPARE(editor->text(), QStringLiteral("features"));
    QVERIFY(editor->toolTip().contains(QStringLiteral("only input slot")));
    QTreeWidgetItem *bindingRow = nullptr;
    for (int i = 0; i < inspector->topLevelItemCount(); ++i) {
        QTreeWidgetItem *parameters = inspector->topLevelItem(i);
        if (parameters->text(0) != QStringLiteral("Parameters")) continue;
        for (int j = 0; j < parameters->childCount(); ++j)
            if (parameters->child(j)->text(0) == QStringLiteral("binding"))
                bindingRow = parameters->child(j);
    }
    QVERIFY(bindingRow && bindingRow->childCount() == 1);
    QVERIFY(bindingRow->child(0)->text(1).contains(QStringLiteral("only input slot")));

    QTimer::singleShot(0, &window, [&window, &app, &binding, &bindingEditor] {
        auto *manager = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(manager);
        auto *list = manager->findChild<QListWidget *>("datasetManagerList");
        auto *activate = manager->findChild<QPushButton *>("datasetManagerActivate");
        QVERIFY(list && activate);
        auto selectDataset = [list](const QString &id) {
            for (int i = 0; i < list->count(); ++i) {
                if (list->item(i)->data(Qt::UserRole).toString() == id) {
                    list->setCurrentItem(list->item(i));
                    return true;
                }
            }
            return false;
        };
        QVERIFY(selectDataset(QStringLiteral("auto.second")));
        QVERIFY(activate->isEnabled());
        activate->click();
        QTRY_COMPARE(binding(), QStringLiteral("pixels"));
        QTRY_VERIFY(bindingEditor() && bindingEditor()->text() == QStringLiteral("pixels"));
        QVERIFY(bindingEditor()->isReadOnly());

        QVERIFY(selectDataset(QStringLiteral("auto.multiple")));
        activate->click();
        QTRY_COMPARE(QString::fromUtf8(nn_project_active_dataset(nn_app_project(app))->id),
                     QStringLiteral("auto.multiple"));
        QTRY_VERIFY(bindingEditor() && !bindingEditor()->isReadOnly());
        QLineEdit *editor = bindingEditor();
        editor->setText(QStringLiteral("target"));
        QVERIFY(QMetaObject::invokeMethod(editor, "editingFinished", Qt::DirectConnection));
        QTRY_COMPARE(binding(), QStringLiteral("target"));
        manager->reject();
    });
    manage->trigger();
    answerDialog(QMessageBox::Discard);
    window.close();
}

void WindowTest::typedOutputAuthoringBoundariesAndInspector() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "typed-qt", "Typed Qt", "0.1.0", false,
                           error, sizeof(error)), error);
    const QByteArray dataset = R"({"name":"Features","description":"","batch":{"inputs":{"features":{"dtype":"float32","shape":["B",4]}},"targets":{}}})";
    QVERIFY2(nn_app_create_dataset(app, "typed.features", "1.0.0", dataset.constData(), true,
                                   error, sizeof(error)), error);
    MainWindow window(app);
    window.show();
    QTest::qWait(30);
    auto *diagnostics = window.findChild<QTreeWidget *>("diagnostics");
    QVERIFY(diagnostics);
    QTreeWidgetItem *rootProblem = nullptr;
    for (int i = 0; i < diagnostics->topLevelItemCount(); ++i)
        if (diagnostics->topLevelItem(i)->text(1) == QStringLiteral("Root"))
            rootProblem = diagnostics->topLevelItem(i);
    QVERIFY(rootProblem);
    QVERIFY(rootProblem->text(0).contains(QStringLiteral("Incomplete")));
    QVERIFY(rootProblem->data(0, Qt::UserRole).toString().isEmpty()); // not a synthetic node
    QAction *create = nullptr;
    for (QAction *action : window.findChildren<QAction *>())
        if (action->text() == QString::fromUtf8("Create stereotype…")) create = action;
    QVERIFY(create);
    QTimer::singleShot(0, &window, [&window] {
        auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
        QVERIFY(dialog);
        dialog->findChild<QLineEdit *>("stereotypeId")->setText("ui.typed");
        dialog->findChild<QLineEdit *>("stereotypeVersion")->setText("1.0.0");
        dialog->findChild<QLineEdit *>("stereotypeName")->setText("Typed producer");
        auto *overrideBox = dialog->findChild<QCheckBox *>("stereotypeOutputOverride");
        overrideBox->setChecked(true);
        for (QPushButton *button : dialog->findChildren<QPushButton *>())
            if (button->text() == QStringLiteral("Add output row")) {
                button->click(); button->click(); break;
            }
        auto *outputs = dialog->findChild<QTableWidget *>("stereotypeOutputs");
        outputs->item(0, 0)->setText("prediction");
        outputs->item(0, 1)->setText("output");
        outputs->item(1, 0)->setText("objective");
        outputs->item(1, 1)->setText("output"); // retain these values after rejected form
        setInitialStereotypeLua(dialog, QStringLiteral(
            "return function(context, parameters, services)\n"
            "  local input = context.inputs[1]\n"
            "  if not input then return { status = 'unresolved', message = 'missing' } end\n"
            "  return { status = 'success', outputs = { prediction = input, objective = input } }\n"
            "end"));
        auto *save = dialog->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save);
        QTimer::singleShot(0, dialog, [dialog, outputs, save] {
            save->click();
            QTimer::singleShot(0, dialog, [dialog, outputs] {
                QVERIFY(dialog->isVisible());
                QCOMPARE(dialog->findChild<QLineEdit *>("stereotypeId")->text(), QStringLiteral("ui.typed"));
                QCOMPARE(outputs->item(1, 0)->text(), QStringLiteral("objective"));
                QVERIFY(!dialog->findChild<QLabel *>("stereotypeError")->text().isEmpty());
                outputs->item(1, 1)->setText("loss");
                QTimer::singleShot(0, dialog, [dialog] {
                    dialog->findChild<QDialogButtonBox *>()->button(QDialogButtonBox::Save)->click();
                });
            });
        });
    });
    create->trigger();
    const NNPackage *typedPackage = nn_catalog_find(nn_project_catalog(nn_app_project(app)),
                                                    "ui.typed", "1.0.0");
    QVERIFY(typedPackage);
    QCOMPARE(typedPackage->output_count, size_t(2));
    QCOMPARE(QString::fromUtf8(typedPackage->outputs[0].type), QStringLiteral("output"));
    QCOMPARE(QString::fromUtf8(typedPackage->outputs[1].type), QStringLiteral("loss"));

    QVERIFY2(nn_app_add_node(app, "feature-input", "core.input", "0.1.0", "", 20, 150,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_set_parameter_text(app, "feature-input", "binding", "features",
                                       error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "typed-source", "ui.typed", "1.0.0", "", 260, 150,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "consumer-one", "core.relu", "0.1.0", "", 520, 70,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "consumer-two", "core.relu", "0.1.0", "", 520, 180,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "consumer-loss", "core.relu", "0.1.0", "", 520, 290,
                             error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "typed-input-edge", "feature-input", "out", "typed-source", "in",
                            error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "fanout-one", "typed-source", "prediction", "consumer-one", "in",
                            error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "fanout-two", "typed-source", "prediction", "consumer-two", "in",
                            error, sizeof(error)), error);
    QVERIFY2(nn_app_connect(app, "loss-edge", "typed-source", "objective", "consumer-loss", "in",
                            error, sizeof(error)), error);

    auto *scene = window.findChild<GraphScene *>();
    QVERIFY(scene);
    scene->refresh();
    PortItem *prediction = nullptr, *objective = nullptr;
    for (PortItem *port : scene->nodeItem("typed-source")->ports()) {
        if (!port->isOutput()) continue;
        if (port->handleId() == QStringLiteral("prediction")) prediction = port;
        if (port->handleId() == QStringLiteral("objective")) objective = port;
    }
    QVERIFY(prediction && objective);
    QCOMPARE(prediction->outputType(), QStringLiteral("output"));
    QCOMPARE(objective->outputType(), QStringLiteral("loss"));
    QCOMPARE(scene->edgeItem("fanout-one")->pen().color(), QColor("#111111"));
    QCOMPARE(scene->edgeItem("fanout-two")->pen().color(), QColor("#111111"));
    QCOMPARE(scene->edgeItem("loss-edge")->pen().color(), QColor("#c62828"));

    auto *view = window.findChild<GraphView *>();
    QVERIFY(view);
    const QPoint start = view->mapFromScene(objective->scenePos());
    QTest::mousePress(view->viewport(), Qt::LeftButton, Qt::NoModifier, start);
    QTest::mouseMove(view->viewport(), start + QPoint(45, 35), 30);
    QCoreApplication::processEvents();
    QCOMPARE(scene->connectionDraftColor(), QColor("#c62828"));
    QTest::keyClick(view, Qt::Key_Escape);
    QTest::mouseRelease(view->viewport(), Qt::LeftButton, Qt::NoModifier, start + QPoint(45, 35));

    const NNModel *rootModel = nn_app_model(app);
    const NNNode *rootOutput = nullptr;
    for (size_t i = 0; i < nn_model_node_count(rootModel); ++i) {
        const NNNode *candidate = nn_model_node_at(rootModel, i);
        if (candidate && (!candidate->scope_id || !*candidate->scope_id) &&
            std::strcmp(candidate->package_id, "core.output") == 0) rootOutput = candidate;
    }
    QVERIFY(rootOutput);
    NodeItem *terminalItem = scene->nodeItem(QString::fromUtf8(rootOutput->id));
    QVERIFY(terminalItem);
    PortItem *terminalInput = nullptr;
    for (PortItem *port : terminalItem->ports())
        if (!port->isOutput()) terminalInput = port;
    QVERIFY(terminalInput);
    char rejectedError[512] = {};
    QVERIFY(!nn_app_connect(app, "invalid-loss-terminal-edge", "typed-source", "objective",
                            rootOutput->id, "in", rejectedError, sizeof(rejectedError)));
    QVERIFY(*rejectedError);
    scene->errorOccurred(QStringLiteral("Cannot connect nodes: %1").arg(QString::fromUtf8(rejectedError)));
    QCoreApplication::processEvents();
    QMessageBox *rejection = nullptr;
    for (QMessageBox *candidate : window.findChildren<QMessageBox *>())
        if (candidate->isVisible() && candidate->text().startsWith(QStringLiteral("Cannot connect nodes:")))
            rejection = candidate;
    QVERIFY(rejection && !rejection->isModal());
    rejection->close();

    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    QVERIFY2(report, error);
    const NNInferenceResult *typedResult = nullptr;
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (result && result->node_id && std::strcmp(result->node_id, "typed-source") == 0)
            typedResult = result;
    }
    QVERIFY2(typedResult && typedResult->status == NN_INFERENCE_SUCCESS && typedResult->output_count == 2,
             typedResult ? typedResult->message : "No inference result for typed source");
    QCOMPARE(QString::fromUtf8(typedResult->outputs[0].handle_id), QStringLiteral("prediction"));

    scene->nodeItem("typed-source")->setSelected(true);
    QCoreApplication::processEvents();
    auto *inspector = window.findChild<QTreeWidget *>("inspector");
    QVERIFY(inspector);
    QTreeWidgetItem *outputsRow = nullptr;
    for (int i = 0; i < inspector->topLevelItemCount(); ++i)
        if (inspector->topLevelItem(i)->text(0) == QStringLiteral("Successful outputs"))
            outputsRow = inspector->topLevelItem(i);
    QVERIFY(outputsRow);
    QCOMPARE(outputsRow->childCount(), 2);

    // C-spawned subflow terminal mappings are presented in the nested inspector.
    QVERIFY2(nn_app_add_node(app, "mapped-flow", "core.subflow-proxy", "0.1.0", "", 800, 200,
                             error, sizeof(error)), error);
    scene->setScope(QStringLiteral("mapped-flow"));
    scene->refresh();
    const NNModel *model = nn_app_model(app);
    const NNNode *terminal = nullptr;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (node && node->scope_id && std::strcmp(node->scope_id, "mapped-flow") == 0 &&
            std::strcmp(node->package_id, "core.output") == 0) terminal = node;
    }
    QVERIFY(terminal && terminal->boundary_handle_id);
    const QString terminalId = QString::fromUtf8(terminal->id);
    scene->nodeItem(terminalId)->setSelected(true);
    QCoreApplication::processEvents();
    auto *mapping = window.findChild<QComboBox *>("boundaryMapping");
    QVERIFY(mapping);
    QCOMPARE(mapping->currentData().toString(), QStringLiteral("out"));
    answerDialog(QMessageBox::Discard);
    window.close();
}

void WindowTest::automationRejectsInvalidScopeAndCapturesCurrentScope() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "scope-test", "Scope test", "0.1.0", false,
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

    const auto unknownReveal = runCli({"--socket", socketPath, "ui.reveal", "{\"id\":\"missing\"}"});
    QVERIFY(unknownReveal.first != 0);
    QCOMPARE(inspectScope(), QStringLiteral("flow"));
    const auto revealRootNode = runCli({"--socket", socketPath, "ui.reveal", "{\"id\":\"flow\"}"});
    QCOMPARE(revealRootNode.first, 0);
    QCOMPARE(inspectScope(), QString());
    auto *scene = window.findChild<GraphScene *>();
    QVERIFY(scene);
    QTRY_COMPARE(scene->selectedNodeId(), QStringLiteral("flow"));
    const auto revealChild = runCli({"--socket", socketPath, "ui.reveal", "{\"id\":\"child\"}"});
    QCOMPARE(revealChild.first, 0);
    QCOMPARE(inspectScope(), QStringLiteral("flow"));
    QTRY_COMPARE(scene->selectedNodeId(), QStringLiteral("child"));

    const auto enterEncoder = runCli({"--socket", socketPath, "ui.scope", "{\"id\":\"encoder\"}"});
    QCOMPARE(enterEncoder.first, 0);
    QCOMPARE(inspectScope(), QStringLiteral("encoder"));
    const QByteArray vaePath = QByteArray(NN_SOURCE_DIR) + "/examples/models/mnist-vae";
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
    environment.insert(QStringLiteral("NNMODELLING_ACTIVITY_LOG"), QStringLiteral("off"));
    process.setProcessEnvironment(environment);
    process.start();
    QVERIFY(process.waitForStarted(2000));
    QVERIFY(process.waitForFinished(5000));
    QCOMPARE(process.exitCode(), 2);
    QVERIFY(process.readAllStandardError().contains("--socket requires a socket path"));
}

static bool copyDirectory(const QString &from, const QString &to) {
    QDir().mkpath(to);
    QDirIterator it(from, QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    const QDir root(from);
    while (it.hasNext()) {
        const QString source = it.next();
        const QString target = QDir(to).filePath(root.relativeFilePath(source));
        QDir().mkpath(QFileInfo(target).absolutePath());
        if (!QFile::copy(source, target)) return false;
    }
    return true;
}

void WindowTest::activityLogRecordsUiAndCoreEvents() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    const QString project = temporary.filePath(QStringLiteral("project"));
    QVERIFY(copyDirectory(QStringLiteral(NN_SOURCE_DIR "/examples/test"), project));
    const QString logPath = temporary.filePath(QStringLiteral("activity.log"));
    char error[512] = {};
    NNActivityLog *log = nn_activity_open(logPath.toUtf8().constData(), error, sizeof(error));
    QVERIFY2(log, error);
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    nn_app_set_activity_log(app, log);
    ActivityLog::setLog(log);
    {
        MainWindow window(app);
        window.show();
        QVERIFY(QTest::qWaitForWindowExposed(&window));
        QVERIFY(window.openProject(project));
        QCoreApplication::processEvents();

        auto *save = window.findChild<QAction *>(QStringLiteral("saveProjectAction"));
        QVERIFY(save);
        save->trigger();

        auto *palette = window.findChild<QTreeWidget *>(QStringLiteral("palette"));
        auto *add = window.findChild<QPushButton *>(QStringLiteral("addNodeButton"));
        QVERIFY(palette && add);
        QTreeWidgetItemIterator iterator(palette);
        QTreeWidgetItem *relu = nullptr;
        for (; *iterator; ++iterator)
            if ((*iterator)->data(0, Qt::UserRole).toString() == QStringLiteral("core.relu")) relu = *iterator;
        QVERIFY(relu);
        palette->setCurrentItem(relu);
        relu->setSelected(true);
        QTRY_VERIFY(add->isEnabled());
        add->click();

        auto *scene = window.findChild<GraphScene *>();
        auto *view = window.findChild<GraphView *>();
        QVERIFY(scene && view);
        NodeItem *node = nullptr;
        QTRY_VERIFY((node = [&] {
            for (QGraphicsItem *item : scene->items())
                if (auto *candidate = dynamic_cast<NodeItem *>(item)) {
                    const NNNode *model = nn_model_find_node(nn_app_model(app),
                        candidate->id().toUtf8().constData());
                    if (model && QString::fromUtf8(model->package_id) == QStringLiteral("core.relu"))
                        return candidate;
                }
            return static_cast<NodeItem *>(nullptr);
        }()) != nullptr);
        scene->clearSelection();
        node->setSelected(true);
        QCoreApplication::processEvents();
        PortItem *out = nullptr;
        for (PortItem *port : node->ports()) if (port->isOutput()) out = port;
        QVERIFY(out);
        const QPoint start = view->mapFromScene(out->scenePos());
        QTest::mousePress(view->viewport(), Qt::LeftButton, Qt::NoModifier, start);
        QTest::mouseMove(view->viewport(), start + QPoint(0, 120), 30);
        QTest::mouseRelease(view->viewport(), Qt::LeftButton, Qt::NoModifier, start + QPoint(0, 120));
        QCoreApplication::processEvents();
        // The window owns the application; detach the log before it is destroyed.
        nn_app_set_activity_log(app, nullptr);
        ActivityLog::setLog(nullptr);
    }
    nn_activity_close(log);

    QFile file(logPath);
    QVERIFY(file.open(QIODevice::ReadOnly));
    QMap<QString, QJsonObject> first;
    QStringList events;
    QString previous;
    while (!file.atEnd()) {
        QJsonParseError parseError;
        const QByteArray line = file.readLine();
        QVERIFY(line.endsWith('\n'));
        const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
        QVERIFY2(document.isObject(), line.constData());
        const QJsonObject record = document.object();
        QCOMPARE(record.value(QStringLiteral("v")).toInt(), 1);
        const QString event = record.value(QStringLiteral("event")).toString();
        // A successful node.add is followed by its model.snapshot.
        if (previous == QStringLiteral("node.add")) QCOMPARE(event, QStringLiteral("model.snapshot"));
        previous = event;
        events << event;
        if (!first.contains(event)) first.insert(event, record);
    }
    QVERIFY(first.contains(QStringLiteral("project.open")));
    QCOMPARE(first.value(QStringLiteral("ui.action")).value(QStringLiteral("data")).toObject()
                 .value(QStringLiteral("action")).toString(), QStringLiteral("saveProjectAction"));
    QCOMPARE(first.value(QStringLiteral("ui.action")).value(QStringLiteral("data")).toObject()
                 .value(QStringLiteral("via")).toString(), QStringLiteral("menu"));
    // The triggering action precedes the effects of its handler.
    QVERIFY(events.contains(QStringLiteral("project.save")));
    QVERIFY(events.indexOf(QStringLiteral("ui.action")) < events.indexOf(QStringLiteral("project.save")));
    const QJsonObject paletteAdd = first.value(QStringLiteral("palette.add")).value(QStringLiteral("data")).toObject();
    QCOMPARE(paletteAdd.value(QStringLiteral("package")).toString(), QStringLiteral("core.relu"));
    QCOMPARE(paletteAdd.value(QStringLiteral("via")).toString(), QStringLiteral("button"));
    QCOMPARE(first.value(QStringLiteral("node.add")).value(QStringLiteral("src")).toString(), QStringLiteral("core"));
    QVERIFY(events.indexOf(QStringLiteral("palette.add")) < events.indexOf(QStringLiteral("node.add")));
    QVERIFY(events.contains(QStringLiteral("model.snapshot")));
    QVERIFY(events.contains(QStringLiteral("palette.select")));
    const QJsonObject selection = first.value(QStringLiteral("selection.change")).value(QStringLiteral("data")).toObject();
    QVERIFY(selection.contains(QStringLiteral("nodes")));
    QVERIFY(events.contains(QStringLiteral("connect.start")));
    QCOMPARE(first.value(QStringLiteral("connect.cancel")).value(QStringLiteral("data")).toObject()
                 .value(QStringLiteral("reason")).toString(), QStringLiteral("no_target"));
}

QTEST_MAIN(WindowTest)
#include "qt_window_test.moc"
