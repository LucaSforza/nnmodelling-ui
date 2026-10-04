#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "DataflowLayout.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#include <QMessageBox>
#include <QStatusBar>
#include <QByteArray>
#include <cstring>
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

void MainWindow::arrangeCurrentScope(FlowDirection direction) {
    const NNModel *model = nn_app_model(application_.get());
    if (!model) return;
    scene_->setFlowDirection(direction);
    scene_->refresh();
    const QByteArray scope = scene_->scope().toUtf8();
    QVector<DataflowLayout::Node> nodes;
    QVector<DataflowLayout::Edge> edges;
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
        bool useOutputs = false;
        for (PortItem *port : item ? item->ports() : QList<PortItem *>{})
            useOutputs = useOutputs || port->isOutput();
        qreal anchor = direction == FlowDirection::Vertical ? bounds.center().x() : bounds.center().y();
        qreal anchorSum = 0.0;
        int anchorCount = 0;
        if (item) {
            for (PortItem *port : item->ports()) {
                if (port->isOutput() != useOutputs) continue;
                anchorSum += direction == FlowDirection::Vertical ? port->pos().x() : port->pos().y();
                ++anchorCount;
            }
        }
        if (anchorCount) anchor = anchorSum / anchorCount;
        nodes.push_back({QString::fromUtf8(node->id), bounds, anchor, terminal});
    }
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        if (!edge || std::strcmp(edge->scope_id ? edge->scope_id : "", scope.constData()) != 0)
            continue;
        edges.push_back({QString::fromUtf8(edge->source_id), QString::fromUtf8(edge->target_id)});
    }
    const QHash<QString, QPointF> positions = DataflowLayout::arrange(nodes, edges, direction);
    char error[ErrorCapacity] = {};
    if (!nn_app_begin_edit(application_.get(), error, sizeof(error))) {
        QMessageBox::warning(this, tr("Arrange failed"), QString::fromUtf8(error));
        return;
    }
    for (const DataflowLayout::Node &node : nodes) {
        const QPointF position = positions.value(node.id);
        const QByteArray id = node.id.toUtf8();
        if (!nn_app_move_node(application_.get(), id.constData(), position.x(), position.y(),
                              error, sizeof(error))) {
            const QString failure = QString::fromUtf8(error);
            char rollbackError[ErrorCapacity] = {};
            if (!nn_app_end_edit(application_.get(), false, rollbackError, sizeof(rollbackError)))
                QMessageBox::critical(this, tr("Arrange rollback failed"),
                    QStringLiteral("%1\n%2").arg(failure, QString::fromUtf8(rollbackError)));
            else
                QMessageBox::warning(this, tr("Arrange failed"), failure);
            scene_->refresh();
            return;
        }
    }
    if (!nn_app_end_edit(application_.get(), true, error, sizeof(error))) {
        QMessageBox::warning(this, tr("Arrange failed"), QString::fromUtf8(error));
        scene_->refresh();
        return;
    }
    refreshAll();
    view_->fitGraph();
}
