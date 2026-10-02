#include "MainWindow.hpp"
#include "GraphScene.hpp"
#include "GraphView.hpp"
#include "NodeItem.hpp"
#include "MainWindowUtils.hpp"
#include "application/application.h"
#include "model/model.h"
#include <QMessageBox>
#include <QStatusBar>
#include <QByteArray>
#include <QRectF>
#include <algorithm>
#include <cstring>
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
