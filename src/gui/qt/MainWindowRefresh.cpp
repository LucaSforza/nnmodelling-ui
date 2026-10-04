#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "Network3DView.hpp"
#include <QTabWidget>
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#include <QBrush>
#include <QAction>
#include <QColor>
#include <QLineEdit>
#include <QToolButton>
#include <QTreeWidget>
#include <QTreeWidgetItem>
#include <QTreeWidgetItemIterator>
#include <map>
#include <set>
#include <string>
#include <functional>
#include <utility>

using namespace MainWindowUtils;

void MainWindow::refreshAll() {
    if (!scene_) return;
    refreshing_ = true;
    scene_->refresh();
    if (networkTabs_ && networkTabs_->currentWidget() == network3DView_ && network3DView_)
        network3DView_->rebuild(application_ ? nn_app_project(application_.get()) : nullptr);
    QString scope = scene_->scope();
    scopeTree_->clear();
    auto *rootItem = new QTreeWidgetItem(scopeTree_, {tr("Root")});
    rootItem->setData(0, IdRole, QString());
    std::set<std::string> scopes;
    std::map<std::string, QString> scopeLabels;
    std::map<std::string, std::string> parents;
    const NNModel *model = nn_app_model(application_.get());
    for (size_t i = 0; model && i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (!node) continue;
        if (node->scope_id && *node->scope_id) scopes.emplace(node->scope_id);
        if (nn_app_node_is_subflow(application_.get(), node->id)) {
            scopes.emplace(node->id);
            scopeLabels[node->id] = textOr(QString::fromUtf8(node->label), node->id);
            parents[node->id] = node->scope_id ? node->scope_id : "";
        }
    }
    const auto labelFor = [&scopeLabels](const std::string &id) {
        return scopeLabels.count(id) ? scopeLabels.at(id) : QString::fromStdString(id);
    };
    std::set<std::string> reachable;
    std::set<std::string> unresolved;
    for (const std::string &id : scopes) {
        std::set<std::string> chain;
        std::string at = id;
        bool rooted = false;
        while (!at.empty()) {
            if (!chain.insert(at).second) break;
            const auto parent = parents.find(at);
            if (parent == parents.end()) break;
            at = parent->second;
            if (at.empty()) { rooted = true; break; }
        }
        if (rooted) reachable.insert(id);
        else unresolved.insert(id);
    }
    std::function<void(QTreeWidgetItem *, const std::string &, std::set<std::string>)> addChildren;
    addChildren = [&](QTreeWidgetItem *parentItem, const std::string &parentId,
                      std::set<std::string> ancestors) {
        for (const std::string &id : scopes) {
            const auto parent = parents.find(id);
            if (!reachable.count(id) || parent == parents.end() || parent->second != parentId ||
                ancestors.count(id)) continue;
            auto *item = new QTreeWidgetItem(parentItem, {labelFor(id)});
            item->setData(0, IdRole, QString::fromStdString(id));
            auto nextAncestors = ancestors;
            nextAncestors.insert(id);
            addChildren(item, id, std::move(nextAncestors));
        }
    };
    addChildren(rootItem, "", {});
    if (!unresolved.empty()) {
        auto *orphans = new QTreeWidgetItem(scopeTree_, {tr("Orphan scopes")});
        orphans->setData(0, IdRole, QString());
        orphans->setFlags(orphans->flags() & ~Qt::ItemIsSelectable);
        std::set<std::string> placed;
        std::function<void(QTreeWidgetItem *, const std::string &, std::set<std::string>)> addOrphans;
        addOrphans = [&](QTreeWidgetItem *parentItem, const std::string &parentId,
                         std::set<std::string> ancestors) {
            for (const std::string &id : unresolved) {
                const auto parent = parents.find(id);
                if (placed.count(id) || parent == parents.end() || parent->second != parentId ||
                    ancestors.count(id)) continue;
                auto *item = new QTreeWidgetItem(parentItem, {labelFor(id)});
                item->setData(0, IdRole, QString::fromStdString(id));
                placed.insert(id);
                auto nextAncestors = ancestors;
                nextAncestors.insert(id);
                addOrphans(item, id, std::move(nextAncestors));
            }
        };
        // Missing parents start orphan chains. Any remaining entries form a cycle.
        for (const std::string &id : unresolved) {
            const auto parent = parents.find(id);
            if (parent == parents.end() || !unresolved.count(parent->second)) {
                auto *item = new QTreeWidgetItem(orphans, {labelFor(id)});
                item->setData(0, IdRole, QString::fromStdString(id));
                placed.insert(id);
                addOrphans(item, id, {id});
            }
        }
        for (const std::string &id : unresolved) {
            if (placed.count(id)) continue;
            auto *item = new QTreeWidgetItem(orphans, {labelFor(id)});
            item->setData(0, IdRole, QString::fromStdString(id));
            placed.insert(id);
        }
    }
    scopeTree_->expandAll();
    QTreeWidgetItemIterator current(scopeTree_);
    while (*current && (*current)->data(0, IdRole).toString() != scope) ++current;
    if (*current) {
        scopeTree_->setCurrentItem(*current);
        scopeSelector_->setText((*current)->text(0));
    } else {
        scopeTree_->setCurrentItem(rootItem);
        scopeSelector_->setText(rootItem->text(0));
        scene_->setScope(QString());
    }
    refreshing_ = false;
    refreshPalette();
    refreshInspector();
    refreshResources();
    refreshDiagnostics();
    updateWindowTitle();
    undoAction_->setEnabled(nn_app_can_undo(application_.get()));
    redoAction_->setEnabled(nn_app_can_redo(application_.get()));
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
