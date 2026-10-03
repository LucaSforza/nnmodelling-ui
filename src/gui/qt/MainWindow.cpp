#include "MainWindow.hpp"

#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "automation/automation.h"
#include "project/project.h"

#include <QAction>
#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QFileDialog>
#include <QHeaderView>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
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

#include <cmath>

using namespace MainWindowUtils;

namespace {
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

    auto *fileMenu = menuBar()->addMenu(tr("&Project"));
    auto addAction = [this, fileMenu](const QString &label, const QKeySequence &shortcut,
                                      auto callback) {
        QAction *action = fileMenu->addAction(label);
        if (!shortcut.isEmpty()) action->setShortcut(shortcut);
        connect(action, &QAction::triggered, this, callback);
        return action;
    };
    addAction(tr("New project…"), {}, [this] { createProject(false); });
    addAction(tr("New MNIST MLP…"), {}, [this] { createProject(true); });
    addAction(tr("New MNIST VAE…"), {}, [this] { createVaeProject(); });
    fileMenu->addSeparator();
    fileMenu->addAction(tr("Create stereotype…"), this, &MainWindow::createStereotype);
    fileMenu->addAction(tr("Create dataset…"), this, &MainWindow::createDataset);
    addAction(tr("Open project…"), QKeySequence::Open, [this] {
        const QString directory = QFileDialog::getExistingDirectory(this, tr("Open project"));
        if (!directory.isEmpty()) openProject(directory);
    });
    addAction(tr("Save"), QKeySequence::Save, [this] { saveProject(); });
    addAction(tr("Close project"), {}, [this] {
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
    });

    auto *toolbar = addToolBar(tr("Workspace"));
    toolbar->setMovable(false);
    toolbar->setIconSize(QSize(16, 16));
    for (QAction *action : fileMenu->actions()) toolbar->addAction(action);
    toolbar->addSeparator();
    toolbar->addWidget(new QLabel(tr("Scope"), toolbar));
    scopeSelector_ = new QComboBox(toolbar);
    scopeSelector_->setObjectName(QStringLiteral("scopeSelector"));
    scopeSelector_->setMinimumWidth(190);
    toolbar->addWidget(scopeSelector_);
    auto *fitAction = toolbar->addAction(tr("Fit"));
    auto *zoomInAction = toolbar->addAction(tr("+"));
    zoomInAction->setObjectName(QStringLiteral("zoomIn"));
    zoomInAction->setToolTip(tr("Zoom in"));
    auto *zoomOutAction = toolbar->addAction(tr("−"));
    zoomOutAction->setObjectName(QStringLiteral("zoomOut"));
    zoomOutAction->setToolTip(tr("Zoom out"));
    auto *arrangeAction = toolbar->addAction(tr("Arrange"));
    auto *arrangeMenu = new QMenu(this);
    QAction *verticalAction = arrangeMenu->addAction(tr("Vertical"));
    QAction *horizontalAction = arrangeMenu->addAction(tr("Horizontal"));
    if (auto *button = qobject_cast<QToolButton *>(toolbar->widgetForAction(arrangeAction))) {
        button->setMenu(arrangeMenu);
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

    connect(scopeSelector_, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (refreshing_ || index < 0 || !scene_) return;
        scene_->setScope(scopeSelector_->itemData(index).toString());
    });
    connect(scene_, &GraphScene::modelChanged, this, [this] {
        QTimer::singleShot(0, this, [this] { refreshAll(); });
    });
    connect(scene_, &GraphScene::scopeChanged, this, [this](const QString &scope) {
        if (refreshing_) return;
        const int index = scopeSelector_->findData(scope);
        if (index >= 0) {
            const QSignalBlocker blocker(scopeSelector_);
            scopeSelector_->setCurrentIndex(index);
        }
        refreshInspector();
        view_->fitGraph();
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
    connect(fitAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Vertical, false);
    });
    connect(zoomInAction, &QAction::triggered, view_, &GraphView::zoomIn);
    connect(zoomOutAction, &QAction::triggered, view_, &GraphView::zoomOut);
    connect(arrangeAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Vertical);
    });
    connect(verticalAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Vertical);
    });
    connect(horizontalAction, &QAction::triggered, this, [this] {
        arrangeCurrentScope(FlowDirection::Horizontal);
    });
}
