#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "MainWindowUtils.hpp"
#include "PortItem.hpp"
#include "EdgeItem.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "inference/inference.h"
#include "project/project.h"

#include <QAction>
#include <QApplication>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>
#include <QComboBox>
#include <QPushButton>
#include <QMessageBox>
#include <QMenu>
#include <QMenuBar>
#include <QLineF>
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
#include <QTreeWidgetItemIterator>
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
    void stereotypeReferenceForm();
    void modelProblemDiagnosticsAndNavigation();
    void currentScopeRetainsOutsideCauseContext();
    void retainedLuaValidationFormAndNonblockingRejection();
    void visualResourceAuthoringForms();
    void typedOutputAuthoringBoundariesAndInspector();
    void automationRejectsInvalidScopeAndCapturesCurrentScope();
    void arrangeMenuAndFitUseDirectedGridLayout();
    void graphHistoryTracksEditsAndScopeFallback();
    void scopeTreeFollowsContainmentAndMenus();
    void socketOptionRequiresPath();
};

void WindowTest::stereotypeReferenceForm() {
    QTemporaryDir temporary;
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    QVERIFY2(nn_app_create(app, temporary.path().toUtf8().constData(), "reference-form",
                          "Reference form", false, error, sizeof(error)), error);
    QVERIFY2(nn_app_add_node(app, "parallel", "core.horizontal-repeat", "0.1.0", "",
                            0, 0, error, sizeof(error)), error);
    QVERIFY2(nn_app_create_stereotype(app, "form.join", "1.0.0",
        R"({"name":"Form join","kind":"join","view":{"color":"#4779c4","width":180,"height":100},"parameters":{"enabled":{"type":"boolean","default":true},"factor":{"type":"number","default":1.0,"minimum":0},"mode":{"type":"string","default":"sum","choices":["sum","mean"]}}})",
        "return function(c,p,s) return {status='success',output=c.inputs[1]} end",
        "{}", error, sizeof(error)), error);
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
    QCOMPARE(dimension->text(), QStringLiteral("-1"));
    QVERIFY(dimension->validator());
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
    QCoreApplication::processEvents();
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    auto *enabled = window.findChild<QCheckBox *>("stereotypeParameter_join_enabled");
    auto *factor = window.findChild<QLineEdit *>("stereotypeParameter_join_factor");
    auto *mode = window.findChild<QComboBox *>("stereotypeParameter_join_mode");
    QVERIFY(enabled && factor && mode);
    QVERIFY(enabled->isChecked());
    QCOMPARE(mode->count(), 2);
    factor->setText("2.5");
    QVERIFY(QMetaObject::invokeMethod(factor, "editingFinished", Qt::DirectConnection));
    enabled->setChecked(false);
    mode->setCurrentText("mean");
    reference = nn_app_parameter_text(app, "parallel", "join");
    const QJsonObject values = QJsonDocument::fromJson(reference).object().value("parameters").toObject();
    nn_app_free_text(reference);
    QCOMPARE(values.value("factor").toDouble(), 2.5);
    QCOMPARE(values.value("enabled").toBool(), false);
    QCOMPARE(values.value("mode").toString(), QStringLiteral("mean"));
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
    QVERIFY2(nn_app_create(app, parent.constData(), "layout-test", "Layout test", false,
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
    QVERIFY2(nn_app_create(app, parent.constData(), "scope-tree-test", "Scope tree", false,
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
                                   window.menuBar()->actions()[3]->text()};
    QCOMPARE(menuNames, QStringList({QStringLiteral("&File"), QStringLiteral("&Edit"), QStringLiteral("&Model"),
                                     QStringLiteral("&View")}));
    QMenu *templates = window.findChild<QMenu *>("projectTemplatesMenu");
    QVERIFY(templates && templates->actions().size() == 2);
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
    QVERIFY2(nn_app_create(app, parent.constData(), "history-test", "History test", false,
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

void WindowTest::modelProblemDiagnosticsAndNavigation() {
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray example = QByteArray(NN_SOURCE_DIR) + "/examples/models/mnist-mlp";
    QVERIFY2(nn_app_open(app, example.constData(), error, sizeof(error)), error);
    QVERIFY2(nn_app_set_parameter_text(app, "dense1", "in_features", "800",
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

    MainWindow window(app);
    auto *scene = window.findChild<GraphScene *>();
    auto *diagnostics = window.findChild<QTreeWidget *>("diagnostics");
    auto *filter = window.findChild<QCheckBox *>("currentScopeProblems");
    QVERIFY(scene && diagnostics && filter);
    QVERIFY(diagnostics->topLevelItemCount() >= 2);
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
    QCOMPARE(diagnostics->columnCount(), 1);
    QVERIFY(diagnostics->wordWrap());
    const NNNode *denseNode = nn_model_find_node(nn_app_model(app), "dense1");
    QVERIFY(denseNode && denseNode->label);
    const QString denseLabel = QString::fromUtf8(denseNode->label);
    window.show();
    QCoreApplication::processEvents();
    const QString normalCapture = qEnvironmentVariable("NN_PROBLEMS_CAPTURE");
    if (!normalCapture.isEmpty()) QVERIFY(window.grab().save(normalCapture));
    window.resize(900, 560);
    QCoreApplication::processEvents();
    const QString smallCapture = qEnvironmentVariable("NN_PROBLEMS_CAPTURE_SMALL");
    if (!smallCapture.isEmpty()) QVERIFY(window.grab().save(smallCapture));
    const QString fullRow = root->text(0);
    const QStringList rowLines = fullRow.split('\n');
    QVERIFY(fullRow.contains(QStringLiteral("Model error")));
    QVERIFY(fullRow.contains(denseLabel));
    QVERIFY(rowLines.size() >= 3);
    QVERIFY(rowLines[1].startsWith(QStringLiteral("Scope:")));
    QVERIFY(!rowLines[2].trimmed().isEmpty());
    QCOMPARE(root->toolTip(0), fullRow);
    const QRect rootRect = diagnostics->visualItemRect(root);
    QVERIFY(rootRect.isValid());
    QVERIFY(rootRect.height() >= diagnostics->fontMetrics().lineSpacing() * 3);

    scene->setScope(QStringLiteral("flow"));
    filter->setChecked(true);
    QCoreApplication::processEvents();
    QVERIFY(diagnostics->topLevelItemCount() > 0);
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
}

void WindowTest::retainedLuaValidationFormAndNonblockingRejection() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "lua-retained", "Lua retained", false,
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
        dialog->findChild<QPlainTextEdit *>("stereotypeLua")->setPlainText("return function(");
        QTimer::singleShot(0, dialog, [dialog] {
            auto *buttons = dialog->findChild<QDialogButtonBox *>();
            buttons->button(QDialogButtonBox::Save)->click();
            QTimer::singleShot(0, dialog, [dialog] {
                QVERIFY(dialog->isVisible());
                QCOMPARE(dialog->findChild<QLineEdit *>("stereotypeId")->text(), QStringLiteral("bad.lua"));
                auto *luaError = dialog->findChild<QLabel *>("stereotypeLuaError");
                QVERIFY(!luaError->text().isEmpty());
                QVERIFY(!dialog->findChild<QPlainTextEdit *>("stereotypeLua")->extraSelections().isEmpty());
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
    QVERIFY(causeItem->text(0).contains(QStringLiteral("cause context"), Qt::CaseInsensitive));
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

void WindowTest::typedOutputAuthoringBoundariesAndInspector() {
    QTemporaryDir temporary;
    QVERIFY(temporary.isValid());
    NNApplication *app = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
    QVERIFY(app);
    char error[512] = {};
    const QByteArray parent = temporary.path().toUtf8();
    QVERIFY2(nn_app_create(app, parent.constData(), "typed-qt", "Typed Qt", false,
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
        if (diagnostics->topLevelItem(i)->text(0).contains(QStringLiteral("· Root\n")))
            rootProblem = diagnostics->topLevelItem(i);
    QVERIFY(rootProblem);
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
        dialog->findChild<QPlainTextEdit *>("stereotypeLua")->setPlainText(
            "return function(context, parameters, services)\n"
            "  local input = context.inputs[1]\n"
            "  if not input then return { status = 'unresolved', message = 'missing' } end\n"
            "  return { status = 'success', outputs = { prediction = input, objective = input } }\n"
            "end");
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
    process.setProcessEnvironment(environment);
    process.start();
    QVERIFY(process.waitForStarted(2000));
    QVERIFY(process.waitForFinished(5000));
    QCOMPARE(process.exitCode(), 2);
    QVERIFY(process.readAllStandardError().contains("--socket requires a socket path"));
}

QTEST_MAIN(WindowTest)
#include "qt_window_test.moc"
