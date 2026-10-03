#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#include <QMessageBox>
#include <QStatusBar>
#include <QByteArray>
#include <QRectF>
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <QUuid>
#include <QTreeWidget>
#include <QTreeWidgetItem>

using namespace MainWindowUtils;

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

void MainWindow::arrangeCurrentScope(FlowDirection direction, bool frameContent) {
    const NNModel *model = nn_app_model(application_.get());
    if (!model) return;
    scene_->setFlowDirection(direction);
    const QByteArray scope = scene_->scope().toUtf8();
    struct LayoutNode {
        std::string id;
        double width;
        double height;
        bool terminal;
        int rank = 0;
    };
    std::vector<LayoutNode> nodes;
    std::string firstInputId;
    std::map<std::string, size_t> indexes;
    const NNProject *project = nn_app_project(application_.get());
    const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (!node) continue;
        const char *nodeScope = node->scope_id ? node->scope_id : "";
        if (std::strcmp(nodeScope, scope.constData()) != 0) continue;
        const NodeItem *item = scene_->nodeItem(QString::fromUtf8(node->id));
        const QRectF bounds = item ? item->boundingRect() : QRectF(0, 0, 190, 96);
        const NNPackage *package = catalog
            ? nn_catalog_find(catalog, node->package_id, node->package_version) : nullptr;
        const bool terminal = package && package->kind &&
            (std::strcmp(package->kind, "output") == 0 ||
             std::strcmp(package->kind, "loss-output") == 0);
        const bool input = package && package->kind &&
                           std::strcmp(package->kind, "input") == 0;
        if (input && firstInputId.empty()) firstInputId = node->id;
        indexes.emplace(node->id, nodes.size());
        nodes.push_back({node->id, bounds.width(), bounds.height(), terminal});
    }
    std::vector<std::vector<size_t>> next(nodes.size());
    std::vector<size_t> incoming(nodes.size(), 0);
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        if (!edge || std::strcmp(edge->scope_id ? edge->scope_id : "", scope.constData()) != 0)
            continue;
        const auto source = indexes.find(edge->source_id ? edge->source_id : "");
        const auto target = indexes.find(edge->target_id ? edge->target_id : "");
        if (source == indexes.end() || target == indexes.end()) continue;
        next[source->second].push_back(target->second);
        ++incoming[target->second];
    }
    std::set<std::pair<std::string, size_t>> ready;
    for (size_t i = 0; i < nodes.size(); ++i)
        if (incoming[i] == 0) ready.emplace(nodes[i].id, i);
    while (!ready.empty()) {
        const size_t index = ready.begin()->second;
        ready.erase(ready.begin());
        for (size_t target : next[index]) {
            nodes[target].rank = qMax(nodes[target].rank, nodes[index].rank + 1);
            if (--incoming[target] == 0) ready.emplace(nodes[target].id, target);
        }
    }
    int lastComputationRank = 0;
    for (const LayoutNode &node : nodes)
        if (!node.terminal) lastComputationRank = qMax(lastComputationRank, node.rank);
    int lastRank = lastComputationRank;
    for (LayoutNode &node : nodes) {
        if (node.terminal) node.rank = qMax(node.rank, lastComputationRank + 1);
        lastRank = qMax(lastRank, node.rank);
    }
    std::vector<std::vector<size_t>> ranks(size_t(lastRank + 1));
    for (size_t i = 0; i < nodes.size(); ++i) ranks[size_t(nodes[i].rank)].push_back(i);
    for (auto &rank : ranks)
        std::sort(rank.begin(), rank.end(), [&nodes](size_t a, size_t b) {
            return nodes[a].id < nodes[b].id;
        });
    constexpr double grid = NN_MODEL_GRID_SPACING;
    constexpr double margin = 80.0;
    constexpr double gap = 60.0;
    const auto gridCeil = [=](double value) { return std::ceil(value / grid) * grid; };
    std::vector<double> rankFlowSizes(ranks.size(), 0.0);
    for (size_t rank = 0; rank < ranks.size(); ++rank)
        for (size_t index : ranks[rank])
            rankFlowSizes[rank] = qMax(rankFlowSizes[rank],
                direction == FlowDirection::Vertical ? nodes[index].height : nodes[index].width);
    std::vector<double> rankOffsets(ranks.size(), 0.0);
    for (size_t rank = 1; rank < ranks.size(); ++rank)
        rankOffsets[rank] = rankOffsets[rank - 1] + gridCeil(rankFlowSizes[rank - 1]) + gap;
    const double totalFlow = ranks.empty() ? 0.0
        : rankOffsets.back() + gridCeil(rankFlowSizes.back());
    std::vector<QPointF> positions(nodes.size());
    for (size_t rankIndex = 0; rankIndex < ranks.size(); ++rankIndex) {
        const auto &rank = ranks[rankIndex];
        double lanePosition = margin;
        for (size_t index : rank) {
            const LayoutNode &node = nodes[index];
            if (direction == FlowDirection::Vertical) {
                positions[index] = QPointF(lanePosition, margin + rankOffsets[rankIndex]);
                lanePosition += gridCeil(node.width + gap);
            } else {
                const double flowX = margin + totalFlow - rankOffsets[rankIndex] -
                                     gridCeil(rankFlowSizes[rankIndex]);
                positions[index] = QPointF(flowX, lanePosition);
                lanePosition += gridCeil(node.height + gap);
            }
        }
    }
    char error[ErrorCapacity] = {};
    for (size_t i = 0; i < nodes.size(); ++i) {
        const QByteArray id = QByteArray::fromStdString(nodes[i].id);
        if (!nn_app_move_node(application_.get(), id.constData(),
                              positions[i].x(), positions[i].y(),
                              error, sizeof(error))) {
            QMessageBox::warning(this, tr("Arrange failed"), QString::fromUtf8(error));
            scene_->refresh();
            return;
        }
    }
    refreshAll();
    if (frameContent) {
        view_->fitGraph();
    } else if (!firstInputId.empty()) {
        if (NodeItem *input = scene_->nodeItem(QString::fromStdString(firstInputId)))
            view_->focusAtTop(input->sceneBoundingRect());
        else
            view_->fitGraph();
    } else {
        view_->fitGraph();
    }
}
