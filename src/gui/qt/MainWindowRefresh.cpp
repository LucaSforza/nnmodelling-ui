#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#include <QBrush>
#include <QColor>
#include <QComboBox>
#include <QLineEdit>
#include <QSignalBlocker>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <map>
#include <set>
#include <string>

using namespace MainWindowUtils;

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
