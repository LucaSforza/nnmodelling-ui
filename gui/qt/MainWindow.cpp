#include "MainWindow.hpp"

#include "GraphScene.hpp"
#include "GraphView.hpp"

#include "application.h"
#include "catalog.h"
#include "inference.h"
#include "model.h"
#include "project.h"

#include <QAction>
#include <QBrush>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QColor>
#include <QDir>
#include <QFileDialog>
#include <QHeaderView>
#include <QInputDialog>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QPointF>
#include <QSize>
#include <QSplitter>
#include <QStringList>
#include <QStatusBar>
#include <QSignalBlocker>
#include <QTimer>
#include <QTreeWidget>
#include <QToolBar>
#include <QVBoxLayout>
#include <QVariant>
#include <QUuid>
#include <QWidget>

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
constexpr int IdRole = Qt::UserRole;
constexpr int PackageVersionRole = Qt::UserRole + 1;
constexpr size_t ErrorCapacity = 512;

QString textOr(QString value, const char *fallback) {
    return value.isEmpty() ? QString::fromUtf8(fallback) : value;
}

QString packageColor(const NNPackage *package) {
    QString color = package && package->color ? QString::fromUtf8(package->color) : QString();
    if (color.startsWith('#') && (color.size() == 4 || color.size() == 7)) return color;
    return QStringLiteral("#6b8fc4");
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
}

MainWindow::MainWindow(NNApplication *application, QWidget *parent)
    : QMainWindow(parent), application_(application, nn_app_free) {
    buildUi();
    refreshAll();
}

MainWindow::~MainWindow() {
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
        "QTreeWidget, QLineEdit, QComboBox { background: #ffffff; }"
        "QPushButton { padding: 5px 10px; }"
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
    addButton->setProperty("primary", true);
    leftLayout->addWidget(addButton);

    scene_ = new GraphScene(application_.get(), this);
    view_ = new GraphView(scene_, workspace);
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
            [this](QTreeWidgetItem *item, int column) { selectResource(item, column); });
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
    QPushButton *open = chooser.addButton(tr("Open project"), QMessageBox::ActionRole);
    chooser.addButton(QMessageBox::Close);
    chooser.exec();
    if (chooser.clickedButton() == create) createProject(false);
    else if (chooser.clickedButton() == mnist) createProject(true);
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
        item->setForeground(0, QBrush(QColor(packageColor(package))));
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
        item->setForeground(0, QBrush(QColor(packageColor(package))));
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
    std::vector<std::string> ids;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (!node) continue;
        const char *nodeScope = node->scope_id ? node->scope_id : "";
        if (std::strcmp(nodeScope, scope.constData()) == 0) ids.emplace_back(node->id);
    }
    constexpr int columns = 4;
    constexpr double horizontalSpacing = 280.0;
    constexpr double verticalSpacing = 150.0;
    char error[ErrorCapacity] = {};
    for (size_t i = 0; i < ids.size(); ++i) {
        const QByteArray id = QByteArray::fromStdString(ids[i]);
        if (!nn_app_move_node(application_.get(), id.constData(),
                              80.0 + (i % columns) * horizontalSpacing,
                              70.0 + (i / columns) * verticalSpacing,
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
