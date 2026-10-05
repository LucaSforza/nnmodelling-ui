#include "MainWindow.hpp"

#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "MainWindowUtils.hpp"
#include "BackendDialog.hpp"
#include "application/application.h"
#include "automation/automation.h"
#include "project/project.h"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QFileDialog>
#include <QHeaderView>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QHideEvent>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSplitter>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSize>
#include <QStyledItemDelegate>
#include <QStyleOptionViewItem>
#include <QPalette>
#include <QBrush>
#include <QColor>
#include <QTreeView>
#include <QTreeWidgetItemIterator>
#include <QWidgetAction>

#include <cmath>

using namespace MainWindowUtils;

namespace {
class EditorMenu final : public QMenu {
public:
    using QMenu::QMenu;
protected:
    void hideEvent(QHideEvent *event) override {
        QMenu::hideEvent(event);
        if (QGuiApplication::platformName() != QStringLiteral("vnc")) return;
        // Qt VNC miscomposes reused popup surfaces. Replace only the closed
        // container; retain actions (including shortcuts) and embedded trees.
        QTimer::singleShot(0, this, [this] {
            if (isVisible()) return;
            auto *replacement = new EditorMenu(title(), parentWidget());
            replacement->setObjectName(objectName());
            const auto entries = actions();
            for (QAction *action : entries) {
                removeAction(action);
                if (action->parent() == this) action->setParent(replacement);
                if (action->menu() && action->menu()->parent() == this)
                    action->menu()->setParent(replacement, action->menu()->windowFlags());
                replacement->addAction(action);
            }
            const auto owners = menuAction()->associatedObjects();
            for (QObject *owner : owners) {
                if (auto *button = qobject_cast<QToolButton *>(owner)) {
                    button->setMenu(replacement);
                } else if (auto *widget = qobject_cast<QWidget *>(owner)) {
                    widget->insertAction(menuAction(), replacement->menuAction());
                    widget->removeAction(menuAction());
                }
            }
            deleteLater();
        });
    }
};

class WrappedTreeDelegate final : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;
    QSize sizeHint(const QStyleOptionViewItem &option, const QModelIndex &index) const override {
        QStyleOptionViewItem itemOption(option);
        initStyleOption(&itemOption, index);
        const auto *tree = qobject_cast<const QTreeView *>(parent());
        int depth = 0;
        for (QModelIndex parentIndex = index.parent(); parentIndex.isValid(); parentIndex = parentIndex.parent()) ++depth;
        int width = option.rect.width();
        if (width <= 0 && tree) width = tree->viewport()->width();
        if (tree) width -= (depth + 1) * tree->indentation();
        width = qMax(48, width - 24);
        const QRect textBounds = QFontMetrics(itemOption.font).boundingRect(QRect(0, 0, width, 10000), Qt::TextWordWrap, itemOption.text);
        return QSize(width, qMax(28, textBounds.height() + 10));
    }
};
}

MainWindow::MainWindow(NNApplication *application, QWidget *parent)
    : QMainWindow(parent), application_(application, nn_app_free) {
    QPalette palette = QApplication::palette();
    palette.setColor(QPalette::Window, QColor("#eceff3"));
    palette.setColor(QPalette::WindowText, QColor("#202c3b"));
    palette.setColor(QPalette::Base, QColor("#ffffff"));
    palette.setColor(QPalette::AlternateBase, QColor("#f5f7f9"));
    palette.setColor(QPalette::Text, QColor("#202c3b"));
    palette.setColor(QPalette::Button, QColor("#e5e9ee"));
    palette.setColor(QPalette::ButtonText, QColor("#202c3b"));
    palette.setColor(QPalette::BrightText, QColor("#ffffff"));
    palette.setColor(QPalette::Highlight, QColor("#3978c5"));
    palette.setColor(QPalette::HighlightedText, QColor("#ffffff"));
    palette.setColor(QPalette::PlaceholderText, QColor("#667587"));
    palette.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#525f6e"));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#657181"));
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#657181"));
    buildUi();
    setPalette(palette);
    refreshAll();
}

MainWindow::~MainWindow() {
    if (automationTimer_) automationTimer_->stop();
    if (automation_) {
        nn_automation_stop(automation_);
        automation_ = nullptr;
    }
    // Destroy scene items before the application they reference.
    delete view_;
    view_ = nullptr;
    delete scene_;
    scene_ = nullptr;
}

void MainWindow::closeEvent(QCloseEvent *event) {
    const NNProject *project = nn_app_project(application_.get());
    bool discard = false;
    if (project && nn_project_dirty(project)) {
        const auto choice = QMessageBox::question(
            this, tr("Close NNModelling"), tr("Save changes before closing?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);
        if (choice == QMessageBox::Cancel) {
            event->ignore();
            return;
        }
        discard = choice == QMessageBox::Discard;
        if (!discard && !saveProject()) {
            event->ignore();
            return;
        }
    }
    if (project) {
        char error[ErrorCapacity] = {};
        if (!nn_app_close(application_.get(), discard, error, sizeof(error))) {
            QMessageBox::critical(this, tr("Close failed"), QString::fromUtf8(error));
            event->ignore();
            return;
        }
    }
    event->accept();
}

void MainWindow::buildUi() {
    setObjectName(QStringLiteral("main-window"));
    setWindowTitle(QStringLiteral("NNModelling"));
    resize(1360, 850);
    setMinimumSize(900, 560);
    setStyleSheet(QStringLiteral(
        "QMainWindow { background: #eceff3; }"
        "QMenuBar, QToolBar { background: #222a35; color: #edf1f6; border: 0; }"
        "QMenuBar::item:selected, QMenu::item:selected { background: #35465b; }"
        "QToolBar QToolButton { color: #edf1f6; padding: 5px 9px; }"
        "QToolBar QToolButton:hover { background: #35465b; }"
        "QDockWidget, QGroupBox { background: #edf0f4; }"
        "QWidget { color: #263446; }"
        "QTreeWidget, QLineEdit, QComboBox, QPlainTextEdit, QTableWidget { background: #ffffff; color: #202c3b; selection-background-color: #3978c5; selection-color: #ffffff; }"
        "QLineEdit, QComboBox, QPlainTextEdit, QTableWidget { border: 1px solid #9aa8b8; padding: 3px; }"
        "QLineEdit { placeholder-text-color: #667587; }"
        "QLineEdit:disabled, QComboBox:disabled { background: #e2e6eb; color: #525f6e; }"
        "QHeaderView::section { background: #dfe5eb; color: #263446; padding: 4px; border: 1px solid #c0cad4; }"
        "QLabel { color: #263446; }"
        "QMenuBar, QToolBar { color: #edf1f6; }"
        "QToolBar QLabel { color: #edf1f6; }"
        "QMenuBar::item { color: #edf1f6; }"
        "QMenu { background: #ffffff; color: #202c3b; border: 1px solid #9aa8b8; }"
        "QMenu::item:selected { background: #3978c5; color: #ffffff; }"
        "QPushButton:disabled { background: #e1e5e9; color: #657181; border: 1px solid #aeb8c3; }"
        "QPushButton { background: #e5e9ee; color: #202c3b; border: 1px solid #9aa8b8; padding: 5px 10px; }"
        "QPushButton:hover { background: #d8e1ea; color: #202c3b; }"
        "QPushButton[primary=true] { background: #3978c5; color: white; border: 0; }"
        "QStatusBar { background: #e5e9ee; color: #354154; }"));

    auto addMenu = [this](const QString &title) {
        auto *menu = new EditorMenu(title, this);
        menuBar()->addMenu(menu);
        return menu;
    };
    auto *fileMenu = addMenu(tr("&File"));
    auto addAction = [this](QMenu *menu, const QString &label, const QKeySequence &shortcut,
                            auto callback, const char *objectName = nullptr) {
        QAction *action = menu->addAction(label);
        if (!shortcut.isEmpty()) action->setShortcut(shortcut);
        if (objectName) action->setObjectName(QString::fromLatin1(objectName));
        connect(action, &QAction::triggered, this, callback);
        return action;
    };
    addAction(fileMenu, tr("New project…"), QKeySequence::New,
              [this] { createProject(false); }, "newProjectAction");
    auto *templatesMenu = new EditorMenu(tr("New from template"), fileMenu);
    fileMenu->addMenu(templatesMenu);
    templatesMenu->setObjectName(QStringLiteral("projectTemplatesMenu"));
    addAction(templatesMenu, tr("MNIST MLP…"), {}, [this] { createProject(true); });
    addAction(templatesMenu, tr("MNIST VAE…"), {}, [this] { createVaeProject(); });
    fileMenu->addSeparator();
    addAction(fileMenu, tr("Open project…"), QKeySequence::Open, [this] {
        const QString directory = QFileDialog::getExistingDirectory(this, tr("Open project"));
        if (!directory.isEmpty()) openProject(directory);
    }, "openProjectAction");
    addAction(fileMenu, tr("Save"), QKeySequence::Save, [this] { saveProject(); }, "saveProjectAction");
    addAction(fileMenu, tr("Close project"), QKeySequence::Close, [this] {
        if (!application_ || !nn_app_project(application_.get())) return;
        const NNProject *project = nn_app_project(application_.get());
        bool discard = false;
        if (nn_project_dirty(project)) {
            const auto choice = QMessageBox::question(
                this, tr("Close project"), tr("Save changes before closing?"),
                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
                QMessageBox::Save);
            if (choice == QMessageBox::Cancel) return;
            discard = choice == QMessageBox::Discard;
            if (!discard && !saveProject()) return;
        }
        char error[ErrorCapacity] = {};
        if (!nn_app_close(application_.get(), discard, error, sizeof(error))) {
            QMessageBox::critical(this, tr("Close failed"), QString::fromUtf8(error));
            return;
        }
        refreshAll();
        showProjectChooser();
    }, "closeProjectAction");

    auto *editMenu = addMenu(tr("&Edit"));
    undoAction_ = addAction(editMenu, tr("Undo"), QKeySequence::Undo,
                            [this] { restoreEdit(false); }, "undoAction");
    redoAction_ = addAction(editMenu, tr("Redo"), QKeySequence::Redo,
                            [this] { restoreEdit(true); }, "redoAction");

    auto *modelMenu = addMenu(tr("&Model"));
    addAction(modelMenu, tr("Create stereotype…"), {}, [this] { createStereotype(); });
    addAction(modelMenu, tr("Create dataset…"), {}, [this] { createDataset(); });
    QAction *backendPanelAction = addAction(modelMenu, tr("Training backend…"), {}, [this] {
        BackendDialog dialog(application_.get(), [this] { return saveProject(); },
            [this](const QString &directory) { return openProject(directory); }, this);
        dialog.exec();
    }, "backendPanelAction");

    auto *viewMenu = addMenu(tr("&View"));
    addAction(viewMenu, tr("Fit graph"), QKeySequence(QStringLiteral("Ctrl+0")), [this] {
        view_->fitGraph();
    }, "fitGraphAction");
    addAction(viewMenu, tr("Zoom in"), QKeySequence::ZoomIn,
              [this] { view_->zoomIn(); }, "zoomIn");
    addAction(viewMenu, tr("Zoom out"), QKeySequence::ZoomOut,
              [this] { view_->zoomOut(); }, "zoomOut");
    auto *arrangeMenu = new EditorMenu(tr("Arrange"), viewMenu);
    viewMenu->addMenu(arrangeMenu);
    QAction *verticalAction = arrangeMenu->addAction(tr("Vertical"));
    QAction *horizontalAction = arrangeMenu->addAction(tr("Horizontal"));

    auto *toolbar = addToolBar(tr("Workspace"));
    toolbar->setMovable(false);
    toolbar->setIconSize(QSize(16, 16));
    toolbar->addWidget(new QLabel(tr("Scope"), toolbar));
    scopeSelector_ = new QToolButton(toolbar);
    scopeSelector_->setObjectName(QStringLiteral("scopeSelector"));
    scopeSelector_->setText(tr("Root"));
    scopeSelector_->setMinimumWidth(190);
    scopeSelector_->setPopupMode(QToolButton::InstantPopup);
    auto *scopeMenu = new EditorMenu(scopeSelector_);
    scopeTree_ = new QTreeWidget(scopeMenu);
    scopeTree_->setObjectName(QStringLiteral("scopeTree"));
    scopeTree_->setHeaderHidden(true);
    scopeTree_->setRootIsDecorated(true);
    scopeTree_->setIndentation(16);
    scopeTree_->setMinimumWidth(250);
    scopeTree_->setMaximumHeight(360);
    auto *scopeTreeAction = new QWidgetAction(scopeMenu);
    scopeTreeAction->setDefaultWidget(scopeTree_);
    scopeMenu->addAction(scopeTreeAction);
    scopeSelector_->setMenu(scopeMenu);
    toolbar->addWidget(scopeSelector_);
    auto *trainingAction = toolbar->addAction(tr("Training"));
    trainingAction->setObjectName(QStringLiteral("trainingBackendToolbarAction"));
    trainingAction->setToolTip(tr("Open the training backend"));
    connect(trainingAction, &QAction::triggered, backendPanelAction, &QAction::trigger);
    auto *fitButtonAction = toolbar->addAction(tr("Fit"));
    fitButtonAction->setObjectName(QStringLiteral("fitGraph"));
    auto *zoomInButtonAction = toolbar->addAction(tr("+"));
    zoomInButtonAction->setObjectName(QStringLiteral("zoomInButton"));
    zoomInButtonAction->setToolTip(tr("Zoom in"));
    auto *zoomOutButtonAction = toolbar->addAction(tr("−"));
    zoomOutButtonAction->setObjectName(QStringLiteral("zoomOutButton"));
    zoomOutButtonAction->setToolTip(tr("Zoom out"));
    auto *arrangeAction = toolbar->addAction(tr("Arrange"));
    arrangeAction->setObjectName(QStringLiteral("arrange"));
    auto *toolbarArrangeMenu = new EditorMenu(this);
    QAction *toolbarVertical = toolbarArrangeMenu->addAction(tr("Vertical"));
    QAction *toolbarHorizontal = toolbarArrangeMenu->addAction(tr("Horizontal"));
    if (auto *button = qobject_cast<QToolButton *>(toolbar->widgetForAction(arrangeAction))) {
        button->setMenu(toolbarArrangeMenu);
        button->setPopupMode(QToolButton::MenuButtonPopup);
    }

    auto *workspace = new QSplitter(Qt::Horizontal, this);
    auto *left = new QWidget(workspace);
    left->setMinimumWidth(210);
    auto *leftLayout = new QVBoxLayout(left);
    leftLayout->setContentsMargins(8, 8, 4, 8);
    auto *paletteTitle = new QLabel(tr("Packages"), left);
    paletteTitle->setStyleSheet(QStringLiteral("font-weight: 600; color: #37465a;"));
    leftLayout->addWidget(paletteTitle);
    paletteSearch_ = new QLineEdit(left);
    paletteSearch_->setObjectName(QStringLiteral("paletteSearch"));
    paletteSearch_->setPlaceholderText(tr("Search packages"));
    leftLayout->addWidget(paletteSearch_);
    auto *paletteTree = new QTreeWidget(left);
    paletteTree_ = paletteTree;
    paletteTree->setHeaderHidden(true);
    paletteTree->setObjectName(QStringLiteral("palette"));
    paletteTree->setIndentation(13);
    leftLayout->addWidget(paletteTree, 1);
    auto *addButton = new QPushButton(tr("Add to graph"), left);
    addButton->setObjectName(QStringLiteral("addNodeButton"));
    addButton->setProperty("primary", true);
    leftLayout->addWidget(addButton);

    scene_ = new GraphScene(application_.get(), this);
    view_ = new GraphView(scene_, workspace);
    view_->setObjectName(QStringLiteral("graph"));
    connect(fitButtonAction, &QAction::triggered, view_, &GraphView::fitGraph);
    connect(zoomInButtonAction, &QAction::triggered, view_, &GraphView::zoomIn);
    connect(zoomOutButtonAction, &QAction::triggered, view_, &GraphView::zoomOut);
    view_->setMinimumWidth(360);
    auto *right = new QSplitter(Qt::Vertical, workspace);
    right->setMinimumWidth(255);
    auto *inspectorPane = new QWidget(right);
    auto *inspectorLayout = new QVBoxLayout(inspectorPane);
    inspectorLayout->setContentsMargins(6, 8, 8, 4);
    auto *inspectorTitle = new QLabel(tr("Inspector"), inspectorPane);
    inspectorTitle->setStyleSheet(QStringLiteral("font-weight: 600; color: #37465a;"));
    inspectorLayout->addWidget(inspectorTitle);
    inspector_ = new QTreeWidget(inspectorPane);
    inspector_->setObjectName(QStringLiteral("inspector"));
    inspector_->setColumnCount(2);
    inspector_->setHeaderLabels({tr("Property"), tr("Value")});
    inspector_->header()->setStretchLastSection(true);
    inspectorLayout->addWidget(inspector_, 1);

    auto *resourcePane = new QWidget(right);
    auto *resourceLayout = new QVBoxLayout(resourcePane);
    resourceLayout->setContentsMargins(6, 4, 8, 4);
    auto *resourceTitle = new QLabel(tr("Project resources"), resourcePane);
    resourceTitle->setStyleSheet(QStringLiteral("font-weight: 600; color: #37465a;"));
    resourceLayout->addWidget(resourceTitle);
    resources_ = new QTreeWidget(resourcePane);
    resources_->setObjectName(QStringLiteral("resources"));
    resources_->setHeaderHidden(true);
    resourceLayout->addWidget(resources_, 1);
    auto *resourceActions = new QHBoxLayout;
    auto *stereotypeButton = new QPushButton(tr("New stereotype"), resourcePane);
    stereotypeButton->setObjectName(QStringLiteral("createStereotypeButton"));
    auto *datasetButton = new QPushButton(tr("New dataset"), resourcePane);
    datasetButton->setObjectName(QStringLiteral("createDatasetButton"));
    resourceActions->addWidget(stereotypeButton);
    resourceActions->addWidget(datasetButton);
    resourceLayout->addLayout(resourceActions);

    auto *diagnosticPane = new QWidget(right);
    auto *diagnosticLayout = new QVBoxLayout(diagnosticPane);
    diagnosticLayout->setContentsMargins(6, 4, 8, 8);
    auto *diagnosticTitle = new QLabel(tr("Model problems"), diagnosticPane);
    diagnosticTitle->setStyleSheet(QStringLiteral("font-weight: 600; color: #37465a;"));
    diagnosticLayout->addWidget(diagnosticTitle);
    currentScopeProblems_ = new QCheckBox(tr("Current scope only"), diagnosticPane);
    currentScopeProblems_->setObjectName(QStringLiteral("currentScopeProblems"));
    diagnosticLayout->addWidget(currentScopeProblems_);
    diagnostics_ = new QTreeWidget(diagnosticPane);
    diagnostics_->setObjectName(QStringLiteral("diagnostics"));
    diagnostics_->setColumnCount(1);
    diagnostics_->setHeaderLabels({tr("Problem")});
    diagnostics_->setHeaderHidden(true);
    diagnostics_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    diagnostics_->setItemDelegate(new WrappedTreeDelegate(diagnostics_));
    diagnostics_->setWordWrap(true);
    diagnostics_->setTextElideMode(Qt::ElideNone);
    diagnostics_->setUniformRowHeights(false);
    diagnostics_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    diagnosticLayout->addWidget(diagnostics_, 1);

    right->addWidget(inspectorPane);
    right->addWidget(resourcePane);
    right->addWidget(diagnosticPane);
    right->setSizes({280, 210, 190});
    workspace->addWidget(left);
    workspace->addWidget(view_);
    workspace->addWidget(right);
    workspace->setStretchFactor(0, 0);
    workspace->setStretchFactor(1, 1);
    workspace->setStretchFactor(2, 0);
    workspace->setSizes({235, 820, 305});
    setCentralWidget(workspace);

    const auto chooseScope = [this](QTreeWidgetItem *item) {
        if (refreshing_ || !item || !scene_ || !(item->flags() & Qt::ItemIsSelectable)) return;
        scene_->setScope(item->data(0, IdRole).toString());
        scopeSelector_->menu()->close();
    };
    connect(scopeTree_, &QTreeWidget::itemClicked, this, chooseScope);
    connect(scopeTree_, &QTreeWidget::itemActivated, this, chooseScope);
    connect(scene_, &GraphScene::modelChanged, this, [this] {
        QTimer::singleShot(0, this, [this] { refreshAll(); });
    });
    connect(scene_, &GraphScene::scopeChanged, this, [this](const QString &scope) {
        if (refreshing_) return;
        QTreeWidgetItemIterator iterator(scopeTree_);
        while (*iterator) {
            if ((*iterator)->data(0, IdRole).toString() == scope) {
                scopeTree_->setCurrentItem(*iterator);
                scopeSelector_->setText((*iterator)->text(0));
                break;
            }
            ++iterator;
        }
        // Scope items synchronize asynchronously after event handlers return.
        QTimer::singleShot(0, this, [this] {
            refreshInspector();
            view_->fitGraph();
        });
    });
    connect(scene_, &GraphScene::selectionChanged, this, [this] {
        if (!refreshing_) refreshInspector();
    });
    connect(scene_, &GraphScene::errorOccurred, this, [this](const QString &message) {
        auto *box = new QMessageBox(QMessageBox::Warning, tr("Operation rejected"), message,
                                    QMessageBox::Ok, this);
        box->setAttribute(Qt::WA_DeleteOnClose);
        box->setModal(false);
        box->show();
    });
    connect(diagnostics_, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem *item) {
        if (!item) return;
        const QString id = item->data(0, IdRole).toString();
        if (!id.isEmpty()) revealNode(id);
    });
    connect(currentScopeProblems_, &QCheckBox::toggled, this, [this] { refreshDiagnostics(); });
    connect(paletteSearch_, &QLineEdit::textChanged, this, [this] { refreshPalette(); });
    connect(paletteTree, &QTreeWidget::currentItemChanged, this,
            [this](QTreeWidgetItem *current) {
        selectedPaletteId_ = current && current->childCount() == 0
            ? current->data(0, IdRole).toString() : QString();
    });
    connect(paletteTree, &QTreeWidget::itemDoubleClicked, this,
            [this](QTreeWidgetItem *item, int) {
        if (item && item->childCount() == 0) addSelectedPackage();
    });
    connect(addButton, &QPushButton::clicked, this, [this] { addSelectedPackage(); });
    connect(resources_, &QTreeWidget::itemClicked, this,
            [this](QTreeWidgetItem *item, int column) { selectDataset(item, column); });
    connect(stereotypeButton, &QPushButton::clicked, this, &MainWindow::createStereotype);
    connect(datasetButton, &QPushButton::clicked, this, &MainWindow::createDataset);
    connect(arrangeAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Vertical);
    });
    connect(verticalAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Vertical);
    });
    connect(horizontalAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Horizontal);
    });
    connect(toolbarVertical, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Vertical);
    });
    connect(toolbarHorizontal, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Horizontal);
    });
}

void MainWindow::restoreEdit(bool redo) {
    scene_->cancelInteraction();
    char error[ErrorCapacity] = {};
    const bool restored = redo ? nn_app_redo(application_.get(), error, sizeof(error))
                               : nn_app_undo(application_.get(), error, sizeof(error));
    if (!restored) {
        QMessageBox::warning(this, tr("Edit failed"), QString::fromUtf8(error));
        return;
    }
    refreshAll();
}
