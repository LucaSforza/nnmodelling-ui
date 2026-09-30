#include "MainWindow.hpp"

#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"

#include "application.h"
#include "automation.h"
#include "catalog.h"
#include "inference.h"
#include "model.h"
#include "project.h"

#include <QAction>
#include <QApplication>
#include <QBrush>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QColor>
#include <QDir>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QKeySequence>
#include <QLabel>
#include <QIntValidator>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QPointF>
#include <QSize>
#include <QSet>
#include <QSplitter>
#include <QStringList>
#include <QStatusBar>
#include <QSignalBlocker>
#include <QTimer>
#include <QTreeWidget>
#include <QToolBar>
#include <QVBoxLayout>
#include <QVariant>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QHBoxLayout>
#include <QUuid>
#include <QWidget>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
constexpr int IdRole = Qt::UserRole;
constexpr int PackageVersionRole = Qt::UserRole + 1;
constexpr int DatasetVersionRole = Qt::UserRole + 2;
constexpr size_t ErrorCapacity = 512;

QString textOr(QString value, const char *fallback) {
    return value.isEmpty() ? QString::fromUtf8(fallback) : value;
}

QString packageColor(const NNPackage *package) {
    QString color = package && package->color ? QString::fromUtf8(package->color) : QString();
    if (color.startsWith('#') && (color.size() == 4 || color.size() == 7)) return color;
    return QStringLiteral("#6b8fc4");
}

qreal relativeLuminance(const QColor &color) {
    auto channel = [](int value) {
        const qreal c = value / 255.0;
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * channel(color.red()) + 0.7152 * channel(color.green()) +
           0.0722 * channel(color.blue());
}

QColor readablePackageColor(const NNPackage *package) {
    QColor color(packageColor(package));
    while (color.isValid() && (1.05 / (relativeLuminance(color) + 0.05)) < 4.5)
        color = color.darker(110);
    return color;
}

QString diagnosticStatus(NNInferenceStatus status) {
    switch (status) {
    case NN_INFERENCE_SUCCESS: return QStringLiteral("Success");
    case NN_INFERENCE_SEMANTIC_ERROR: return QStringLiteral("Semantic error");
    case NN_INFERENCE_UNRESOLVED: return QStringLiteral("Unresolved");
    case NN_INFERENCE_RUNTIME_FAULT: return QStringLiteral("Runtime fault");
    }
    return QStringLiteral("Unknown");
}

void addTableRow(QTableWidget *table, const QStringList &defaults) {
    const int row = table->rowCount();
    table->insertRow(row);
    for (int column = 0; column < defaults.size(); ++column) {
        auto *cell = new QTableWidgetItem(defaults[column]);
        table->setItem(row, column, cell);
    }
}

QString cellText(const QTableWidget *table, int row, int column) {
    const auto *item = table->item(row, column);
    return item ? item->text().trimmed() : QString();
}

QJsonValue parseDefault(QString text, const QString &type, bool *ok) {
    *ok = true;
    if (type == QStringLiteral("boolean")) {
        if (text == QStringLiteral("true")) return true;
        if (text == QStringLiteral("false")) return false;
        *ok = false; return {};
    }
    if (type == QStringLiteral("integer")) {
        bool parsed = false;
        const qlonglong integer = text.toLongLong(&parsed);
        *ok = parsed;
        return *ok ? QJsonValue(qint64(integer)) : QJsonValue();
    }
    if (type == QStringLiteral("number")) {
        bool parsed = false;
        const double number = text.toDouble(&parsed);
        *ok = parsed && std::isfinite(number);
        return *ok ? QJsonValue(number) : QJsonValue();
    }
    if (type == QStringLiteral("json")) {
        const auto doc = QJsonDocument::fromJson(text.toUtf8());
        *ok = doc.isArray();
        return *ok ? QJsonValue(doc.array()) : QJsonValue();
    }
    return text;
}
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

bool MainWindow::startAutomation(const QString &socketPath) {
    if (automation_) return true;
    const QByteArray path = QDir::cleanPath(socketPath).toUtf8();
    char error[ErrorCapacity] = {};
    automation_ = nn_automation_start(application_.get(), path.constData(),
                                      &MainWindow::automationUiCallback, this,
                                      error, sizeof(error));
    if (!automation_) {
        statusBar()->showMessage(tr("Automation service failed: %1").arg(QString::fromUtf8(error)));
        return false;
    }
    automationTimer_ = new QTimer(this);
    automationTimer_->setInterval(15);
    connect(automationTimer_, &QTimer::timeout, this, [this] {
        if (!automation_) return;
        const NNProject *previousProject = nn_app_project(application_.get());
        const bool changed = nn_automation_poll(automation_);
        const NNProject *activeProject = nn_app_project(application_.get());
        if (previousProject != activeProject) {
            scene_->clearSelection();
            scene_->setScope(QString());
        }
        if (changed) refreshAll();
    });
    automationTimer_->start();
    statusBar()->showMessage(tr("Automation socket listening: %1").arg(socketPath), 6000);
    return true;
}

char *MainWindow::automationUiCallback(void *user, const char *operation,
                                       const char *argsJson, char *error, size_t cap) {
    auto *self = static_cast<MainWindow *>(user);
    auto fail = [error, cap](const char *message) -> char * {
        if (error && cap) std::snprintf(error, cap, "%s", message);
        return nullptr;
    };
    if (!self || !operation) return fail("Invalid UI callback");
    QJsonParseError parseError{};
    const QJsonDocument argsDoc = QJsonDocument::fromJson(QByteArray(argsJson ? argsJson : "{}"), &parseError);
    if (parseError.error != QJsonParseError::NoError || !argsDoc.isObject())
        return fail("UI arguments must be a JSON object");
    QJsonObject result;
    const QString op = QString::fromUtf8(operation);
    if (op == QStringLiteral("ui.inspect")) {
        QJsonArray widgets;
        struct WidgetInfo { QString id; QString role; QString label; };
        const QList<WidgetInfo> entries = {
            {QStringLiteral("main-window"), QStringLiteral("main-window"), self->windowTitle()},
            {QStringLiteral("palette"), QStringLiteral("package-palette"), self->tr("Packages")},
            {QStringLiteral("graph"), QStringLiteral("graph-canvas"), self->tr("Graph canvas")},
            {QStringLiteral("inspector"), QStringLiteral("node-inspector"), self->tr("Inspector")},
            {QStringLiteral("resources"), QStringLiteral("project-resources"), self->tr("Project resources")},
            {QStringLiteral("diagnostics"), QStringLiteral("diagnostics"), self->tr("Diagnostics")},
            {QStringLiteral("scopeSelector"), QStringLiteral("scope-selector"), self->tr("Scope")},
            {QStringLiteral("createStereotypeButton"), QStringLiteral("resource-action"), self->tr("New stereotype")},
            {QStringLiteral("createDatasetButton"), QStringLiteral("resource-action"), self->tr("New dataset")}};
        for (const auto &entry : entries) {
            QWidget *widget = entry.id == QStringLiteral("main-window")
                ? static_cast<QWidget *>(self) : self->findChild<QWidget *>(entry.id);
            if (!widget) continue;
            QJsonObject item;
            item.insert(QStringLiteral("id"), entry.id);
            item.insert(QStringLiteral("role"), entry.role);
            item.insert(QStringLiteral("label"), widget->accessibleName().isEmpty()
                ? entry.label : widget->accessibleName());
            widgets.append(item);
        }
        result.insert(QStringLiteral("widgets"), widgets);
        result.insert(QStringLiteral("currentScope"), self->scene_ ? self->scene_->scope() : QString());
    } else if (op == QStringLiteral("ui.scope")) {
        const QJsonValue requestedScope = argsDoc.object().value(QStringLiteral("id"));
        if (!requestedScope.isString()) return fail("Scope ID must be a string");
        const QString id = requestedScope.toString();
        if (id.isEmpty()) self->scene_->setScope(QString());
        else {
            const int index = self->scopeSelector_->findData(id);
            if (index < 0) return fail("Unknown scope ID");
            self->scene_->setScope(id);
        }
        self->refreshAll();
        result.insert(QStringLiteral("scope"), self->scene_->scope());
    } else if (op == QStringLiteral("ui.arrange")) {
        self->arrangeCurrentScope();
        result.insert(QStringLiteral("arranged"), true);
    } else if (op == QStringLiteral("ui.screenshot")) {
        const QString path = argsDoc.object().value(QStringLiteral("path")).toString();
        if (path.isEmpty()) return fail("Screenshot path is required");
        self->refreshAll();
        if (self->centralWidget() && self->centralWidget()->layout())
            self->centralWidget()->layout()->activate();
        self->layout()->activate();
        QApplication::processEvents();
        if (!self->grab().save(path)) return fail("Could not save screenshot");
        result.insert(QStringLiteral("path"), path);
        result.insert(QStringLiteral("saved"), true);
    } else {
        return fail("Unsupported UI operation");
    }
    const QByteArray encoded = QJsonDocument(result).toJson(QJsonDocument::Compact);
    char *owned = static_cast<char *>(std::malloc(size_t(encoded.size()) + 1));
    if (!owned) return fail("Out of memory creating UI result");
    std::memcpy(owned, encoded.constData(), size_t(encoded.size()));
    owned[encoded.size()] = '\0';
    return owned;
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
    auto *arrangeAction = toolbar->addAction(tr("Arrange"));

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
    auto *diagnosticTitle = new QLabel(tr("Diagnostics"), diagnosticPane);
    diagnosticTitle->setStyleSheet(QStringLiteral("font-weight: 600; color: #37465a;"));
    diagnosticLayout->addWidget(diagnosticTitle);
    diagnostics_ = new QTreeWidget(diagnosticPane);
    diagnostics_->setObjectName(QStringLiteral("diagnostics"));
    diagnostics_->setColumnCount(2);
    diagnostics_->setHeaderLabels({tr("Result"), tr("Detail")});
    diagnostics_->header()->setStretchLastSection(true);
    diagnostics_->setHeaderHidden(true);
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
        statusBar()->showMessage(message, 8000);
    });
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
    connect(fitAction, &QAction::triggered, view_, &GraphView::fitGraph);
    connect(arrangeAction, &QAction::triggered, this, [this] { arrangeCurrentScope(); });
}

bool MainWindow::openProject(const QString &directory) {
    if (!confirmReplaceProject()) return false;
    const QByteArray path = QDir::cleanPath(directory).toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_open(application_.get(), path.constData(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Open failed"), QString::fromUtf8(error));
        return false;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Opened %1").arg(directory), 4000);
    return true;
}

bool MainWindow::confirmReplaceProject() {
    const NNProject *project = nn_app_project(application_.get());
    if (!project || !nn_project_dirty(project)) return true;
    const auto choice = QMessageBox::question(
        this, tr("Unsaved changes"), tr("Save changes before replacing this project?"),
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
    if (choice == QMessageBox::Cancel) return false;
    return choice == QMessageBox::Discard || saveProject();
}

bool MainWindow::saveProject() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("No project is open"), 4000);
        return false;
    }
    char error[ErrorCapacity] = {};
    if (!nn_app_save(application_.get(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Save failed"), QString::fromUtf8(error));
        return false;
    }
    updateWindowTitle();
    statusBar()->showMessage(tr("Project saved"), 3000);
    return true;
}

void MainWindow::showProjectChooser() {
    QMessageBox chooser(this);
    chooser.setWindowTitle(tr("NNModelling"));
    chooser.setText(tr("Choose a project to edit."));
    QPushButton *create = chooser.addButton(tr("New project"), QMessageBox::ActionRole);
    QPushButton *mnist = chooser.addButton(tr("New MNIST MLP"), QMessageBox::ActionRole);
    QPushButton *vae = chooser.addButton(tr("New MNIST VAE"), QMessageBox::ActionRole);
    QPushButton *open = chooser.addButton(tr("Open project"), QMessageBox::ActionRole);
    chooser.addButton(QMessageBox::Close);
    chooser.exec();
    if (chooser.clickedButton() == create) createProject(false);
    else if (chooser.clickedButton() == mnist) createProject(true);
    else if (chooser.clickedButton() == vae) createVaeProject();
    else if (chooser.clickedButton() == open) {
        const QString directory = QFileDialog::getExistingDirectory(this, tr("Open project"));
        if (!directory.isEmpty()) openProject(directory);
    }
}

void MainWindow::createProject(bool mnist) {
    if (!confirmReplaceProject()) return;
    const QString parent = QFileDialog::getExistingDirectory(
        this, tr("Choose project parent directory"));
    if (parent.isEmpty()) return;
    bool accepted = false;
    const QString id = QInputDialog::getText(this, tr("Project identity"),
        tr("Model ID (used as the new directory name):"), QLineEdit::Normal,
        mnist ? QStringLiteral("mnist-mlp") : QString(), &accepted).trimmed();
    if (!accepted || id.isEmpty()) return;
    const QString name = QInputDialog::getText(this, tr("Project name"), tr("Display name:"),
        QLineEdit::Normal, mnist ? QStringLiteral("MNIST MLP") : id, &accepted).trimmed();
    if (!accepted || name.isEmpty()) return;
    const QByteArray parentPath = QDir::cleanPath(parent).toUtf8();
    const QByteArray projectId = id.toUtf8();
    const QByteArray projectName = name.toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_create(application_.get(), parentPath.constData(), projectId.constData(),
                       projectName.constData(), mnist, error, sizeof(error))) {
        QMessageBox::critical(this, tr("Create failed"), QString::fromUtf8(error));
        return;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Created %1").arg(name), 4000);
}

void MainWindow::createVaeProject() {
    if (!confirmReplaceProject()) return;
    const QString parent = QFileDialog::getExistingDirectory(this, tr("Choose project parent directory"));
    if (parent.isEmpty()) return;
    bool accepted = false;
    const QString id = QInputDialog::getText(this, tr("Project identity"), tr("Model ID:"),
        QLineEdit::Normal, QStringLiteral("mnist-vae"), &accepted).trimmed();
    if (!accepted || id.isEmpty()) return;
    const QString name = QInputDialog::getText(this, tr("Project name"), tr("Display name:"),
        QLineEdit::Normal, QStringLiteral("MNIST VAE"), &accepted).trimmed();
    if (!accepted || name.isEmpty()) return;
    const QByteArray parentBytes = QDir::cleanPath(parent).toUtf8();
    const QByteArray idBytes = id.toUtf8();
    const QByteArray nameBytes = name.toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_create_vae(application_.get(), parentBytes.constData(), idBytes.constData(),
                           nameBytes.constData(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Create VAE failed"), QString::fromUtf8(error));
        return;
    }
    scene_->clearSelection();
    scene_->setScope(QString());
    refreshAll();
    view_->fitGraph();
    statusBar()->showMessage(tr("Created %1").arg(name), 4000);
}

void MainWindow::createStereotype() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before creating a stereotype"), 5000);
        return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Create stereotype — saves project"));
    dialog.setMinimumWidth(760);
    auto *layout = new QVBoxLayout(&dialog);
    auto *form = new QFormLayout;
    auto *id = new QLineEdit(&dialog); id->setObjectName("stereotypeId");
    auto *version = new QLineEdit(QStringLiteral("1.0.0"), &dialog); version->setObjectName("stereotypeVersion");
    auto *name = new QLineEdit(&dialog); name->setObjectName("stereotypeName");
    auto *description = new QLineEdit(&dialog); description->setObjectName("stereotypeDescription");
    auto *kind = new QComboBox(&dialog); kind->setObjectName("stereotypeKind");
    kind->addItems({"input", "layer", "join", "output", "subflow"});
    kind->setCurrentText(QStringLiteral("layer"));
    auto *color = new QLineEdit(QStringLiteral("#6b8fc4"), &dialog); color->setObjectName("stereotypeColor");
    form->addRow(tr("ID"), id); form->addRow(tr("Version"), version);
    form->addRow(tr("Name"), name); form->addRow(tr("Description"), description);
    form->addRow(tr("Kind"), kind); form->addRow(tr("Color"), color);
    layout->addLayout(form);

    layout->addWidget(new QLabel(tr("Parameters — types: boolean, integer, number, string, dtype or JSON array; position: top, bottom or blank."), &dialog));
    auto *parameters = new QTableWidget(0, 7, &dialog);
    parameters->setObjectName("stereotypeParameters");
    parameters->setHorizontalHeaderLabels({tr("Key"), tr("Type"), tr("Default"), tr("Minimum"), tr("Choices (comma separated)"), tr("Position"), tr(" ")});
    parameters->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    parameters->horizontalHeader()->setStretchLastSection(true);
    layout->addWidget(parameters, 1);
    auto *parameterActions = new QHBoxLayout;
    auto *addParameter = new QPushButton(tr("Add parameter row"), &dialog);
    auto *removeParameter = new QPushButton(tr("Remove selected row"), &dialog);
    parameterActions->addWidget(addParameter); parameterActions->addWidget(removeParameter);
    parameterActions->addStretch(); layout->addLayout(parameterActions);
    connect(addParameter, &QPushButton::clicked, &dialog, [parameters] {
        addTableRow(parameters, {QString(), QStringLiteral("number"), QStringLiteral("0"), QString(), QString(), QString(), QString()});
    });
    connect(removeParameter, &QPushButton::clicked, &dialog, [parameters] {
        if (parameters->currentRow() >= 0) parameters->removeRow(parameters->currentRow());
    });

    layout->addWidget(new QLabel(tr("Dependencies — package ID and version constraint."), &dialog));
    auto *dependencies = new QTableWidget(0, 2, &dialog);
    dependencies->setObjectName("stereotypeDependencies");
    dependencies->setHorizontalHeaderLabels({tr("Package ID"), tr("Version constraint")});
    dependencies->horizontalHeader()->setStretchLastSection(true);
    layout->addWidget(dependencies);
    auto *dependencyActions = new QHBoxLayout;
    auto *addDependency = new QPushButton(tr("Add dependency"), &dialog);
    auto *removeDependency = new QPushButton(tr("Remove dependency"), &dialog);
    dependencyActions->addWidget(addDependency); dependencyActions->addWidget(removeDependency);
    dependencyActions->addStretch(); layout->addLayout(dependencyActions);
    connect(addDependency, &QPushButton::clicked, &dialog, [dependencies] {
        addTableRow(dependencies, {QString(), QStringLiteral("0.1.0")});
    });
    connect(removeDependency, &QPushButton::clicked, &dialog, [dependencies] {
        if (dependencies->currentRow() >= 0) dependencies->removeRow(dependencies->currentRow());
    });

    layout->addWidget(new QLabel(tr("Optional Lua inference (pass-through shape rule by default)."), &dialog));
    auto *lua = new QPlainTextEdit(&dialog);
    lua->setObjectName("stereotypeLua");
    const QString passThroughRule = QStringLiteral(
        "return function(context, parameters, services)\n"
        "  if not context.inputs[1] then\n"
        "    return { status = 'unresolved', message = 'Input missing' }\n"
        "  end\n"
        "  return { status = 'success', output = context.inputs[1] }\n"
        "end");
    lua->setPlaceholderText(passThroughRule);
    lua->setPlainText(passThroughRule);
    lua->setMinimumHeight(90);
    layout->addWidget(lua);
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(tr("Create and save project"));
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) return;

    QJsonObject def;
    def.insert("name", name->text().trimmed()); def.insert("description", description->text().trimmed());
    def.insert("kind", kind->currentText());
    QJsonObject view;
    view.insert("color", color->text().trimmed());
    view.insert("width", 190);
    view.insert("height", 110);
    def.insert("view", view);
    QJsonObject params;
    const QSet<QString> parameterTypes = {QStringLiteral("boolean"), QStringLiteral("integer"),
        QStringLiteral("number"), QStringLiteral("string"), QStringLiteral("dtype"),
        QStringLiteral("json")};
    for (int row = 0; row < parameters->rowCount(); ++row) {
        const QString key = cellText(parameters, row, 0);
        const QString type = cellText(parameters, row, 1);
        if (key.isEmpty() || params.contains(key)) {
            QMessageBox::warning(this, tr("Invalid parameter"), tr("Parameter keys must be nonempty and unique.")); return;
        }
        if (!parameterTypes.contains(type)) {
            QMessageBox::warning(this, tr("Invalid parameter type"), tr("Choose boolean, integer, number, string, dtype or JSON array.")); return;
        }
        QJsonObject parameter; parameter.insert("type", type);
        bool validDefault = false;
        const QJsonValue defaultValue = parseDefault(cellText(parameters, row, 2), type, &validDefault);
        if (!validDefault) { QMessageBox::warning(this, tr("Invalid default"), tr("Default value does not match the selected type.")); return; }
        parameter.insert("default", defaultValue);
        const QString minimum = cellText(parameters, row, 3);
        if (!minimum.isEmpty()) {
            bool ok = false; const double value = minimum.toDouble(&ok);
            if (!ok || !std::isfinite(value)) { QMessageBox::warning(this, tr("Invalid minimum"), tr("Minimum must be a finite number.")); return; }
            parameter.insert("minimum", value);
        }
        const QString choices = cellText(parameters, row, 4);
        if (!choices.isEmpty()) { QJsonArray array; for (const QString &choice : choices.split(',', Qt::SkipEmptyParts)) array.append(choice.trimmed()); parameter.insert("choices", array); }
        const QString position = cellText(parameters, row, 5);
        if (!position.isEmpty()) parameter.insert("position", position);
        params.insert(key, parameter);
    }
    def.insert("parameters", params);
    QJsonObject deps;
    for (int row = 0; row < dependencies->rowCount(); ++row) {
        const QString package = cellText(dependencies, row, 0), constraint = cellText(dependencies, row, 1);
        if (package.isEmpty() || constraint.isEmpty() || deps.contains(package)) {
            QMessageBox::warning(this, tr("Invalid dependency"), tr("Dependency IDs and constraints must be nonempty and unique.")); return;
        }
        deps.insert(package, constraint);
    }
    const QByteArray idBytes = id->text().trimmed().toUtf8();
    const QByteArray versionBytes = version->text().trimmed().toUtf8();
    const QByteArray definition = QJsonDocument(def).toJson(QJsonDocument::Compact);
    const QByteArray dependencyJson = QJsonDocument(deps).toJson(QJsonDocument::Compact);
    const QString luaText = lua->toPlainText();
    const QByteArray luaBytes = (luaText.trimmed().isEmpty() ? passThroughRule : luaText).toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_create_stereotype(application_.get(), idBytes.constData(), versionBytes.constData(),
            definition.constData(), luaBytes.constData(), dependencyJson.constData(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Stereotype creation failed"), QString::fromUtf8(error)); return;
    }
    refreshAll();
    statusBar()->showMessage(tr("Stereotype created; current project saved"), 6000);
}

void MainWindow::createDataset() {
    if (!nn_app_project(application_.get())) {
        statusBar()->showMessage(tr("Open a project before creating a dataset"), 5000); return;
    }
    QDialog dialog(this);
    dialog.setWindowTitle(tr("Create dataset — saves project"));
    auto *layout = new QVBoxLayout(&dialog);
    auto *form = new QFormLayout;
    auto *id = new QLineEdit(&dialog); id->setObjectName("datasetId");
    auto *version = new QLineEdit(QStringLiteral("1.0.0"), &dialog); version->setObjectName("datasetVersion");
    auto *name = new QLineEdit(&dialog); name->setObjectName("datasetName");
    auto *description = new QLineEdit(&dialog); description->setObjectName("datasetDescription");
    auto *select = new QCheckBox(tr("Select this dataset after creation"), &dialog);
    select->setObjectName("datasetSelect"); select->setChecked(true);
    form->addRow(tr("ID"), id); form->addRow(tr("Version"), version);
    form->addRow(tr("Name"), name); form->addRow(tr("Description"), description);
    form->addRow(QString(), select); layout->addLayout(form);
    auto makeSlots = [&dialog, layout](const QString &title, const QString &objectName) {
    layout->addWidget(new QLabel(title + tr(" — shape dimensions comma separated (B allowed); dtypes: float16, bfloat16, float32/64, int8/16/32/64, uint8, bool."), &dialog));
        auto *table = new QTableWidget(0, 3, &dialog); table->setObjectName(objectName);
        table->setHorizontalHeaderLabels({tr("Slot name"), tr("Dtype"), tr("Shape")});
        table->horizontalHeader()->setStretchLastSection(true); layout->addWidget(table);
        auto *actions = new QHBoxLayout; auto *add = new QPushButton(tr("Add row"), &dialog);
        auto *remove = new QPushButton(tr("Remove row"), &dialog); actions->addWidget(add); actions->addWidget(remove); actions->addStretch(); layout->addLayout(actions);
        QObject::connect(add, &QPushButton::clicked, table, [table] { addTableRow(table, {QString(), QStringLiteral("float32"), QStringLiteral("B")}); });
        QObject::connect(remove, &QPushButton::clicked, table, [table] { if (table->currentRow() >= 0) table->removeRow(table->currentRow()); });
        return table;
    };
    auto *inputs = makeSlots(tr("Input slots"), QStringLiteral("datasetInputs"));
    auto *targets = makeSlots(tr("Target slots"), QStringLiteral("datasetTargets"));
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    buttons->button(QDialogButtonBox::Save)->setText(tr("Create and save project"));
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject); layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) return;
    auto buildSlots = [this](QTableWidget *table, QJsonObject *out) -> bool {
        static const QSet<QString> dtypes = {"float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"};
        for (int row = 0; row < table->rowCount(); ++row) {
            const QString slot = cellText(table, row, 0), dtype = cellText(table, row, 1);
            if (slot.isEmpty() || out->contains(slot) || !dtypes.contains(dtype)) return false;
            QJsonArray shape;
            const QStringList dims = cellText(table, row, 2).split(',', Qt::KeepEmptyParts);
            if (dims.isEmpty() || dims.size() > 64) return false;
            for (QString dim : dims) {
                dim = dim.trimmed();
                if (dim == QStringLiteral("B")) { shape.append(dim); continue; }
                bool ok = false; const qlonglong n = dim.toLongLong(&ok);
                if (!ok || n <= 0) return false;
                shape.append(QJsonValue(qint64(n)));
            }
            QJsonObject tensor; tensor.insert("dtype", dtype); tensor.insert("shape", shape); out->insert(slot, tensor);
        }
        return true;
    };
    QJsonObject inputSlots, targetSlots;
    if (!buildSlots(inputs, &inputSlots) || inputSlots.isEmpty() || !buildSlots(targets, &targetSlots)) {
        QMessageBox::warning(this, tr("Invalid tensor slots"), tr("Input slots are required; use unique names, supported dtypes and positive dimensions (or B).")); return;
    }
    QJsonObject batch; batch.insert("inputs", inputSlots); batch.insert("targets", targetSlots);
    QJsonObject definition; definition.insert("name", name->text().trimmed());
    definition.insert("description", description->text().trimmed()); definition.insert("batch", batch);
    const QByteArray idBytes = id->text().trimmed().toUtf8(), versionBytes = version->text().trimmed().toUtf8();
    const QByteArray json = QJsonDocument(definition).toJson(QJsonDocument::Compact);
    char error[ErrorCapacity] = {};
    if (!nn_app_create_dataset(application_.get(), idBytes.constData(), versionBytes.constData(),
                              json.constData(), select->isChecked(), error, sizeof(error))) {
        QMessageBox::critical(this, tr("Dataset creation failed"), QString::fromUtf8(error)); return;
    }
    refreshAll();
    statusBar()->showMessage(tr("Dataset created; current project saved"), 6000);
}

void MainWindow::refreshAll() {
    if (!scene_) return;
    refreshing_ = true;
    scene_->refresh();
    const NNProject *project = nn_app_project(application_.get());
    QString scope = scene_->scope();
    const QSignalBlocker scopeBlocker(scopeSelector_);
    scopeSelector_->clear();
    scopeSelector_->addItem(tr("Root"), QString());
    std::set<std::string> scopes;
    std::map<std::string, QString> scopeLabels;
    if (project) {
        const NNModel *model = nn_app_model(application_.get());
        for (size_t i = 0; model && i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node) continue;
            if (node->scope_id && *node->scope_id) scopes.emplace(node->scope_id);
            if (nn_app_node_is_subflow(application_.get(), node->id)) {
                scopes.emplace(node->id);
                scopeLabels[node->id] = textOr(QString::fromUtf8(node->label), node->id);
            }
        }
    }
    for (const std::string &id : scopes) {
        QString label = scopeLabels.count(id) ? scopeLabels[id] : QString::fromStdString(id);
        scopeSelector_->addItem(label, QString::fromStdString(id));
    }
    int scopeIndex = scopeSelector_->findData(scope);
    if (scopeIndex < 0) scopeIndex = 0;
    scopeSelector_->setCurrentIndex(scopeIndex);
    if (scopeSelector_->itemData(scopeIndex).toString() != scope) {
        scene_->setScope(scopeSelector_->itemData(scopeIndex).toString());
    }
    refreshing_ = false;
    refreshPalette();
    refreshInspector();
    refreshResources();
    refreshDiagnostics();
    updateWindowTitle();
}

void MainWindow::refreshPalette() {
    auto *tree = qobject_cast<QTreeWidget *>(paletteTree_);
    if (!tree) return;
    const QString query = paletteSearch_->text().trimmed();
    const NNProject *project = nn_app_project(application_.get());
    const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
    tree->clear();
    selectedPaletteId_.clear();
    if (!catalog) return;
    std::map<QString, QTreeWidgetItem *> groups;
    for (size_t i = 0; i < nn_catalog_count(catalog); ++i) {
        const NNPackage *package = nn_catalog_at(catalog, i);
        if (!package || !package->id || !package->version) continue;
        const QString id = QString::fromUtf8(package->id);
        const QString version = QString::fromUtf8(package->version);
        const QString name = textOr(QString::fromUtf8(package->name), package->id);
        const QString kind = textOr(QString::fromUtf8(package->kind), "Other");
        const QString needle = (name + ' ' + id + ' ' + version + ' ' + kind).toCaseFolded();
        if (!query.isEmpty() && !needle.contains(query.toCaseFolded())) continue;
        auto *group = groups[kind];
        if (!group) {
            group = new QTreeWidgetItem(tree, {kind});
            group->setExpanded(true);
            group->setForeground(0, QBrush(QColor("#58677a")));
            groups[kind] = group;
        }
        auto *item = new QTreeWidgetItem(group, {name});
        item->setData(0, IdRole, id);
        item->setData(0, PackageVersionRole, version);
        item->setToolTip(0, QStringLiteral("%1@%2\n%3").arg(id, version,
            package->description ? QString::fromUtf8(package->description) : QString()));
        item->setForeground(0, QBrush(readablePackageColor(package)));
    }
    tree->expandAll();
}

void MainWindow::refreshInspector() {
    if (!inspector_ || refreshing_) return;
    refreshing_ = true;
    inspector_->clear();
    const QString selectedId = scene_ ? scene_->selectedNodeId() : QString();
    const NNModel *model = nn_app_model(application_.get());
    if (selectedId.isEmpty() || !model) {
        refreshing_ = false;
        return;
    }
    const QByteArray idBytes = selectedId.toUtf8();
    const NNNode *node = nn_model_find_node(model, idBytes.constData());
    if (!node) {
        refreshing_ = false;
        return;
    }
    const QString nodeId = QString::fromUtf8(node->id);
    auto *nameRow = new QTreeWidgetItem(inspector_, {tr("Name"), QString::fromUtf8(node->label)});
    auto *nameEdit = new QLineEdit(QString::fromUtf8(node->label), inspector_);
    inspector_->setItemWidget(nameRow, 1, nameEdit);
    connect(nameEdit, &QLineEdit::editingFinished, this, [this, nameEdit, nodeId] {
        editNodeName(nodeId, nameEdit->text());
    });

    const NNProject *project = nn_app_project(application_.get());
    const NNPackage *package = project
        ? nn_catalog_find(nn_project_catalog(project), node->package_id, node->package_version)
        : nullptr;
    auto *packageRow = new QTreeWidgetItem(inspector_, {tr("Package"),
        QStringLiteral("%1@%2").arg(QString::fromUtf8(node->package_id),
                                     QString::fromUtf8(node->package_version))});
    packageRow->setToolTip(1, package && package->description
        ? QString::fromUtf8(package->description) : QString());
    if (package) {
        auto *parameters = new QTreeWidgetItem(inspector_, {tr("Parameters"), QString()});
        parameters->setExpanded(true);
        for (size_t i = 0; i < package->parameter_count; ++i) {
            const NNParameterDef &definition = package->parameters[i];
            if (!definition.key) continue;
            const QString key = QString::fromUtf8(definition.key);
            char *valueText = nn_app_parameter_text(application_.get(), node->id, definition.key);
            const QString value = valueText ? QString::fromUtf8(valueText) : QString();
            nn_app_free_text(valueText);
            auto *row = new QTreeWidgetItem(parameters, {key, value});
            QWidget *editor = nullptr;
            const QString type = definition.type ? QString::fromUtf8(definition.type) : QString();
            if (definition.choice_count > 0) {
                auto *combo = new QComboBox(inspector_);
                for (size_t j = 0; j < definition.choice_count; ++j) {
                    const QString choice = QString::fromUtf8(definition.choices[j]);
                    combo->addItem(choice);
                }
                combo->setCurrentText(value);
                connect(combo, &QComboBox::currentTextChanged, this,
                        [this, key, nodeId](const QString &next) {
                    editNodeParameter(nodeId, key, next);
                });
                editor = combo;
            } else if (type == QStringLiteral("stereotype")) {
                auto *unsupported = new QLabel(tr("Unsupported native value"), inspector_);
                unsupported->setToolTip(value);
                editor = unsupported;
            } else if (type == QStringLiteral("boolean") || type == QStringLiteral("bool")) {
                auto *check = new QCheckBox(inspector_);
                check->setChecked(value == QStringLiteral("true"));
                connect(check, &QCheckBox::toggled, this, [this, key, nodeId](bool checked) {
                    editNodeParameter(nodeId, key, checked ? QStringLiteral("true")
                                                           : QStringLiteral("false"));
                });
                editor = check;
            } else if (type == QStringLiteral("integer")) {
                auto *line = new QLineEdit(value, inspector_);
                connect(line, &QLineEdit::editingFinished, this, [this, key, nodeId, line] {
                    editNodeParameter(nodeId, key, line->text());
                });
                editor = line;
            } else if (type == QStringLiteral("number") || type == QStringLiteral("real")) {
                auto *line = new QLineEdit(value, inspector_);
                connect(line, &QLineEdit::editingFinished, this, [this, key, nodeId, line] {
                    editNodeParameter(nodeId, key, line->text());
                });
                editor = line;
            } else {
                auto *line = new QLineEdit(value, inspector_);
                connect(line, &QLineEdit::editingFinished, this, [this, key, nodeId, line] {
                    editNodeParameter(nodeId, key, line->text());
                });
                editor = line;
            }
            inspector_->setItemWidget(row, 1, editor);
        }
    }
    inspector_->expandAll();
    refreshing_ = false;
}

void MainWindow::refreshResources() {
    resources_->clear();
    const NNProject *project = nn_app_project(application_.get());
    if (!project) return;
    auto *datasets = new QTreeWidgetItem(resources_, {tr("Datasets")});
    datasets->setExpanded(true);
    for (size_t i = 0; i < nn_project_dataset_count(project); ++i) {
        const NNDataset *dataset = nn_project_dataset_at(project, i);
        if (!dataset) continue;
        const QString identity = QStringLiteral("%1@%2").arg(
            QString::fromUtf8(dataset->id), QString::fromUtf8(dataset->version));
        auto *datasetItem = new QTreeWidgetItem(datasets,
            {QStringLiteral("%1  %2").arg(textOr(QString::fromUtf8(dataset->name), dataset->id), identity)});
        datasetItem->setData(0, IdRole, identity);
        datasetItem->setData(0, PackageVersionRole, QString::fromUtf8(dataset->id));
        datasetItem->setData(0, DatasetVersionRole, QString::fromUtf8(dataset->version));
        if (nn_project_active_dataset(project) == dataset)
            datasetItem->setText(0, QStringLiteral("✓  %1").arg(datasetItem->text(0)));
        datasetItem->setToolTip(0, dataset->path ? QString::fromUtf8(dataset->path) : QString());
        for (size_t j = 0; j < dataset->input_count; ++j) {
            const NNTensorSlot &slot = dataset->inputs[j];
            new QTreeWidgetItem(datasetItem, {QStringLiteral("Input: %1 [%2]").arg(
                QString::fromUtf8(slot.name), QString::fromUtf8(slot.dtype))});
        }
        for (size_t j = 0; j < dataset->target_count; ++j) {
            const NNTensorSlot &slot = dataset->targets[j];
            new QTreeWidgetItem(datasetItem, {QStringLiteral("Target: %1 [%2]").arg(
                QString::fromUtf8(slot.name), QString::fromUtf8(slot.dtype))});
        }
    }
    auto *packages = new QTreeWidgetItem(resources_, {tr("Packages")});
    packages->setExpanded(true);
    const NNCatalog *catalog = nn_project_catalog(project);
    for (size_t i = 0; catalog && i < nn_catalog_count(catalog); ++i) {
        const NNPackage *package = nn_catalog_at(catalog, i);
        if (!package) continue;
        auto *item = new QTreeWidgetItem(packages, {QStringLiteral("%1@%2").arg(
            package->id ? QString::fromUtf8(package->id) : QString(),
            package->version ? QString::fromUtf8(package->version) : QString())});
        item->setForeground(0, QBrush(readablePackageColor(package)));
        for (size_t j = 0; j < package->dependency_count; ++j) {
            const NNPackageDependency &dependency = package->dependencies[j];
            new QTreeWidgetItem(item, {QStringLiteral("Requires %1 %2").arg(
                QString::fromUtf8(dependency.id), QString::fromUtf8(dependency.version_constraint))});
        }
    }
    resources_->expandAll();
}

void MainWindow::refreshDiagnostics() {
    diagnostics_->clear();
    const NNProject *project = nn_app_project(application_.get());
    if (!project) return;
    NNInferenceReport *report = nn_infer_project(project);
    for (size_t i = 0; report && i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (!result) continue;
        const QString node = result->node_id ? QString::fromUtf8(result->node_id) : tr("Project");
        const QString status = diagnosticStatus(result->status);
        QString detail = result->message ? QString::fromUtf8(result->message) : QString();
        if (result->dtype) {
            QStringList dimensions;
            for (size_t j = 0; j < result->dimension_count; ++j)
                dimensions.push_back(QString::fromUtf8(result->dimensions[j]));
            detail += QStringLiteral("  %1[%2]").arg(QString::fromUtf8(result->dtype),
                                                         dimensions.join(QStringLiteral(", ")));
        }
        auto *item = new QTreeWidgetItem(diagnostics_,
            {QStringLiteral("%1  ·  %2").arg(node, status), detail});
        item->setForeground(0, result->status == NN_INFERENCE_SUCCESS
            ? QBrush(QColor("#327450")) : QBrush(QColor("#9a5a26")));
    }
    nn_inference_free(report);
}

void MainWindow::updateWindowTitle() {
    const NNProject *project = nn_app_project(application_.get());
    if (!project) {
        setWindowTitle(tr("NNModelling"));
        return;
    }
    const QString marker = nn_project_dirty(project) ? QStringLiteral(" *") : QString();
    setWindowTitle(QStringLiteral("%1%2 — NNModelling").arg(
        QString::fromUtf8(nn_project_name(project)), marker));
}

void MainWindow::addSelectedPackage() {
    if (!nn_app_project(application_.get()) || selectedPaletteId_.isEmpty()) return;
    auto *tree = qobject_cast<QTreeWidget *>(paletteTree_);
    QTreeWidgetItem *item = tree ? tree->currentItem() : nullptr;
    if (!item || item->childCount() != 0) return;
    const QString packageId = item->data(0, IdRole).toString();
    const QString version = item->data(0, PackageVersionRole).toString();
    const QByteArray id = packageId.toUtf8();
    const QByteArray versionBytes = version.toUtf8();
    const QByteArray scope = scene_->scope().toUtf8();
    const QString nodeId = QStringLiteral("node-%1").arg(
        QUuid::createUuid().toString(QUuid::WithoutBraces));
    const QByteArray nodeIdBytes = nodeId.toUtf8();
    const QPointF center = view_->mapToScene(view_->viewport()->rect().center());
    char error[ErrorCapacity] = {};
    if (!nn_app_add_node(application_.get(), nodeIdBytes.constData(), id.constData(),
                         versionBytes.constData(), scope.constData(), center.x(), center.y(),
                         error, sizeof(error))) {
        QMessageBox::critical(this, tr("Add node failed"), QString::fromUtf8(error));
        return;
    }
    refreshAll();
    statusBar()->showMessage(tr("Added %1").arg(packageId), 3000);
}

void MainWindow::arrangeCurrentScope() {
    const NNModel *model = nn_app_model(application_.get());
    if (!model) return;
    const QByteArray scope = scene_->scope().toUtf8();
    struct LayoutNode { std::string id; double width; double height; };
    std::vector<LayoutNode> nodes;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (!node) continue;
        const char *nodeScope = node->scope_id ? node->scope_id : "";
        if (std::strcmp(nodeScope, scope.constData()) != 0) continue;
        const NodeItem *item = scene_->nodeItem(QString::fromUtf8(node->id));
        const QRectF bounds = item ? item->boundingRect() : QRectF(0, 0, 190, 96);
        nodes.push_back({node->id, bounds.width(), bounds.height()});
    }
    constexpr size_t columns = 4;
    constexpr double gap = 45.0;
    const size_t rowCount = (nodes.size() + columns - 1) / columns;
    std::vector<double> columnWidths(columns, 0.0), rowHeights(rowCount, 0.0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        columnWidths[i % columns] = qMax(columnWidths[i % columns], nodes[i].width);
        rowHeights[i / columns] = qMax(rowHeights[i / columns], nodes[i].height);
    }
    std::vector<double> columnX(columns, 80.0), rowY(rowCount, 70.0);
    for (size_t i = 1; i < columns; ++i)
        columnX[i] = columnX[i - 1] + columnWidths[i - 1] + gap;
    for (size_t i = 1; i < rowCount; ++i)
        rowY[i] = rowY[i - 1] + rowHeights[i - 1] + gap;
    char error[ErrorCapacity] = {};
    for (size_t i = 0; i < nodes.size(); ++i) {
        const QByteArray id = QByteArray::fromStdString(nodes[i].id);
        if (!nn_app_move_node(application_.get(), id.constData(),
                              columnX[i % columns], rowY[i / columns],
                              error, sizeof(error))) {
            QMessageBox::warning(this, tr("Arrange failed"), QString::fromUtf8(error));
            scene_->refresh();
            return;
        }
    }
    refreshAll();
    view_->fitGraph();
}

void MainWindow::editNodeName(const QString &nodeId, const QString &name) {
    if (refreshing_ || !scene_) return;
    const QByteArray id = nodeId.toUtf8();
    const QByteArray label = name.trimmed().toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_rename_node(application_.get(), id.constData(), label.constData(),
                            error, sizeof(error))) {
        QMessageBox::warning(this, tr("Rename failed"), QString::fromUtf8(error));
    }
    QTimer::singleShot(0, this, [this] { refreshAll(); });
}

void MainWindow::editNodeParameter(const QString &nodeId, const QString &key,
                                   const QString &value) {
    if (refreshing_ || !scene_) return;
    const QByteArray id = nodeId.toUtf8();
    const QByteArray keyBytes = key.toUtf8();
    const QByteArray text = value.toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_set_parameter_text(application_.get(), id.constData(), keyBytes.constData(),
                                   text.constData(), error, sizeof(error))) {
        QMessageBox::warning(this, tr("Parameter rejected"), QString::fromUtf8(error));
        return;
    }
    QTimer::singleShot(0, this, [this] { refreshAll(); });
}

void MainWindow::selectResource(QTreeWidgetItem *item, int) {
    if (!item || item->data(0, IdRole).toString().isEmpty()) return;
    // Resource identities are shown here; selection never changes project ownership.
    statusBar()->showMessage(item->toolTip(0), 5000);
}

void MainWindow::selectDataset(QTreeWidgetItem *item, int column) {
    if (!item) return;
    if (item->data(0, PackageVersionRole).toString().isEmpty()) {
        selectResource(item, column);
        return;
    }
    const QByteArray id = item->data(0, PackageVersionRole).toString().toUtf8();
    const QByteArray version = item->data(0, DatasetVersionRole).toString().toUtf8();
    char error[ErrorCapacity] = {};
    if (!nn_app_select_dataset(application_.get(), id.constData(), version.constData(), error, sizeof(error))) {
        QMessageBox::warning(this, tr("Dataset selection failed"), QString::fromUtf8(error));
        return;
    }
    refreshAll();
    statusBar()->showMessage(tr("Selected dataset %1@%2").arg(QString::fromUtf8(id), QString::fromUtf8(version)), 5000);
}
