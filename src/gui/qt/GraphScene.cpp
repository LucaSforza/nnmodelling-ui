#include "GraphScene.hpp"

#include "EdgeItem.hpp"
#include "OrthogonalRouter.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "SubflowItem.hpp"

#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"

#include <QGraphicsPathItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QPainterPath>
#include <QPainter>
#include <QMenu>
#include <QPen>
#include <QSet>
#include <QSignalBlocker>
#include <QVector>
#include <QUuid>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
struct NodeSnapshot {
    QString id;
    QString label;
    QString packageId;
    QString scopeId;
    QString kind;
    QPointF position;
    bool subflow = false;
};
struct EdgeSnapshot {
    QString id;
    QString source;
    QString sourceHandle;
    QString target;
    QString targetHandle;
    QString scopeId;
};

QString copyText(const char *text) { return QString::fromUtf8(text ? text : ""); }
QColor packageColor(const NNApplication *application, const QString &packageId,
                    const QString &version) {
    const NNProject *project = nn_app_project(application);
    const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
    const QByteArray id = packageId.toUtf8();
    const QByteArray packageVersion = version.toUtf8();
    const NNPackage *package = catalog
        ? nn_catalog_find(catalog, id.constData(), packageVersion.constData()) : nullptr;
    const QColor declared = package && package->color ? QColor(QString::fromUtf8(package->color))
                                                       : QColor();
    if (declared.isValid()) return declared;
    const uint h = qHash(packageId);
    return QColor::fromHsv(202 + int(h % 22), 72 + int((h >> 5) % 18),
                           177 + int((h >> 9) % 26));
}
quint64 inputNumber(const QString &handle) {
    bool ok = false;
    const quint64 number = handle.startsWith(QStringLiteral("in-"))
        ? handle.mid(3).toULongLong(&ok) : 0;
    return ok ? number : 0;
}
void sortInputHandles(QStringList *handles) {
    std::sort(handles->begin(), handles->end(), [](const QString &left, const QString &right) {
        const quint64 a = inputNumber(left), b = inputNumber(right);
        return a == b ? left < right : a < b;
    });
}
QString canvasParameterValue(const NNParameterDef &definition, const char *text,
                             const NNCatalog *catalog, QString *error) {
    const QString value = copyText(text);
    if (!definition.type || std::strcmp(definition.type, "stereotype") != 0) return value;

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(value.toUtf8(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        *error = QStringLiteral("Could not format resolved package reference");
        return {};
    }
    const QJsonObject reference = document.object();
    const QByteArray id = reference.value(QStringLiteral("id")).toString().toUtf8();
    const QByteArray version = reference.value(QStringLiteral("version")).toString().toUtf8();
    const NNPackage *package = catalog
        ? nn_catalog_resolve(catalog, id.constData(), version.constData()) : nullptr;
    if (!package || !package->name || !*package->name) {
        *error = QStringLiteral("Resolved package is not in the active catalog");
        return {};
    }
    return copyText(package->name);
}
}

GraphScene::GraphScene(NNApplication *application, QObject *parent)
    : QGraphicsScene(parent), application_(application) {
    setBackgroundBrush(QColor(250, 251, 253));
    setItemIndexMethod(QGraphicsScene::BspTreeIndex);
    setSceneRect(-5000, -5000, 10000, 10000);
    refresh();
}

void GraphScene::drawBackground(QPainter *painter, const QRectF &rect) {
    QGraphicsScene::drawBackground(painter, rect);
    qreal viewScale = 0.0;
    for (QGraphicsView *view : views())
        viewScale = qMax(viewScale, qMin(qAbs(view->transform().m11()),
                                        qAbs(view->transform().m22())));
    if (!(viewScale > 0.0)) viewScale = 1.0;
    constexpr qreal baseSpacing = NN_MODEL_GRID_SPACING;
    const qreal desiredSpacing = qMax<qreal>(10.0 / viewScale,
        qMax(rect.width(), rect.height()) / 300.0);
    const qreal spacing = baseSpacing * qMax<qreal>(1.0,
        std::ceil(desiredSpacing / baseSpacing));
    const qint64 left = qint64(std::floor(rect.left() / spacing));
    const qint64 right = qint64(std::ceil(rect.right() / spacing));
    const qint64 top = qint64(std::floor(rect.top() / spacing));
    const qint64 bottom = qint64(std::ceil(rect.bottom() / spacing));
    QPainterPath grid;
    for (qint64 x = left; x <= right; ++x) {
        grid.moveTo(x * spacing, rect.top());
        grid.lineTo(x * spacing, rect.bottom());
    }
    for (qint64 y = top; y <= bottom; ++y) {
        grid.moveTo(rect.left(), y * spacing);
        grid.lineTo(rect.right(), y * spacing);
    }
    painter->save();
    painter->setPen(QPen(QColor(224, 229, 235), 0));
    painter->drawPath(grid);
    painter->restore();
}

void GraphScene::refresh() {
    if (!application_ || refreshing_) return;
    const void *activeProject = nn_app_project(application_);
    if (projectIdentity_ != activeProject) {
        projectIdentity_ = activeProject;
        flowDirection_ = FlowDirection::Vertical;
        explicitJoinInputHandles_.clear();
        occupiedJoinInputs_.clear();
    }
    refreshing_ = true;
    QSignalBlocker signalBlocker(this);
    dragPositions_.clear();
    QSet<QString> selectedNodes;
    QSet<QString> selectedEdges;
    for (QGraphicsItem *item : selectedItems()) {
        if (auto *node = dynamic_cast<NodeItem *>(item)) selectedNodes.insert(node->id());
        else if (auto *edge = dynamic_cast<EdgeItem *>(item)) selectedEdges.insert(edge->id());
    }

    QVector<NodeSnapshot> nodeData;
    QVector<EdgeSnapshot> edgeData;
    occupiedJoinInputs_.clear();
    const NNModel *model = nn_app_model(application_);
    if (model) {
        nodeData.reserve(int(nn_model_node_count(model)));
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *node = nn_model_node_at(model, i);
            if (!node) continue;
            nodeData.push_back({copyText(node->id), copyText(node->label), copyText(node->package_id),
                                copyText(node->scope_id), {}, QPointF(node->x, node->y), false});
            nodeData.back().subflow = nn_app_node_is_subflow(application_, node->id);
            const NNPackage *package = nn_catalog_find(nn_project_catalog(nn_app_project(application_)),
                                                        node->package_id, node->package_version);
            nodeData.back().kind = package ? copyText(package->kind) : QString();
        }
        edgeData.reserve(int(nn_model_edge_count(model)));
        for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
            const NNEdge *edge = nn_model_edge_at(model, i);
            if (!edge) continue;
            edgeData.push_back({copyText(edge->id), copyText(edge->source_id),
                                copyText(edge->source_handle_id), copyText(edge->target_id),
                                copyText(edge->target_handle_id), copyText(edge->scope_id)});
            occupiedJoinInputs_[copyText(edge->target_id)].append(copyText(edge->target_handle_id));
        }
    }

    clear();
    nodes_.clear();
    edges_.clear();
    nodeScopes_.clear();
    edgeScopes_.clear();
    draftSource_ = nullptr;
    draftPath_ = nullptr;
    QHash<QString, QString> visibleScopes;
    QHash<QString, QStringList> parameterDiagnostics;
    auto addNode = [&](const NodeSnapshot &data, const QPointF &position) {
        if (nodes_.contains(data.id)) return;
        const NNNode *node = nn_model_find_node(nn_app_model(application_), data.id.toUtf8().constData());
        if (!node) return;
        const QColor color = packageColor(application_, data.packageId, copyText(node->package_version));
        QList<QPair<QString, QString>> topParameters, bottomParameters;
        const NNProject *project = nn_app_project(application_);
        const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
        const QByteArray packageId = data.packageId.toUtf8();
        const NNPackage *package = catalog
            ? nn_catalog_find(catalog, packageId.constData(), node->package_version) : nullptr;
        QStringList diagnostics;
        for (size_t i = 0; package && i < package->parameter_count; ++i) {
            const NNParameterDef &definition = package->parameters[i];
            if (!definition.position || !definition.key) continue;
            const QByteArray id = data.id.toUtf8();
            char error[512] = {};
            char *resolved = nn_app_resolved_parameter_text(application_, id.constData(),
                definition.key, error, sizeof(error));
            const bool resolvedSuccessfully = resolved != nullptr;
            QString displayError;
            QString value = resolved
                ? canvasParameterValue(definition, resolved, catalog, &displayError) : QString();
            nn_app_free_text(resolved);
            if (!resolvedSuccessfully || !displayError.isEmpty()) {
                value = QStringLiteral("?");
                const QString reason = displayError.isEmpty() ? copyText(error) : displayError;
                diagnostics.append(QStringLiteral("%1: %2")
                    .arg(QString::fromUtf8(definition.key), reason));
            }
            auto &rows = std::strcmp(definition.position, "top") == 0 ? topParameters : bottomParameters;
            rows.append(qMakePair(QString::fromUtf8(definition.key), value));
        }
        if (!diagnostics.isEmpty()) parameterDiagnostics.insert(data.id, diagnostics);
        NodeItem *item = data.subflow
            ? static_cast<NodeItem *>(new SubflowItem(this, data.id, data.label, data.packageId,
                                                       position, color, topParameters, bottomParameters))
            : new NodeItem(this, data.id, data.label, data.packageId, position, color,
                            topParameters, bottomParameters);
        if (data.kind == QStringLiteral("input") || data.kind == QStringLiteral("output") ||
            data.kind == QStringLiteral("loss-output")) item->setBoundaryKind(data.kind);
        const bool join = data.kind == QStringLiteral("join");
        item->setJoinNode(join);
        item->setFlowDirection(flowDirection_);
        nodes_.insert(data.id, item);
        visibleScopes.insert(data.id, data.scopeId);
        nodeScopes_.insert(data.id, data.scopeId);
        item->setProblemCategory(problemCategories_.value(data.id));
        addItem(item);
        for (bool output : {true, false}) {
            const QByteArray id = data.id.toUtf8();
            QStringList handles;
            if (output) {
                for (size_t index = 0; package && index < package->output_count; ++index)
                    if (package->outputs[index].id) handles.append(QString::fromUtf8(package->outputs[index].id));
            } else if (join) {
                const QStringList occupied = occupiedJoinInputs_.value(data.id);
                handles = {QStringLiteral("in-1"), QStringLiteral("in-2")};
                for (const QString &handle : explicitJoinInputHandles_.value(data.id))
                    if (!handles.contains(handle)) handles.append(handle);
                for (const QString &handle : occupied)
                    if (!handles.contains(handle)) handles.append(handle);
                sortInputHandles(&handles);
            } else {
                const size_t count = nn_app_port_count(application_, id.constData(), false);
                for (size_t index = 0; index < count; ++index) {
                    char handle[128] = {};
                    if (nn_app_port_id(application_, id.constData(), false, index, handle, sizeof(handle)))
                        handles.append(QString::fromUtf8(handle));
                }
            }
            for (const QString &handle : handles) {
                QString type;
                if (output) {
                    const NNOutputDef *definition = nullptr;
                    for (size_t i = 0; package && i < package->output_count; ++i)
                        if (package->outputs[i].id && handle == QString::fromUtf8(package->outputs[i].id))
                            definition = &package->outputs[i];
                    if (!definition) continue;
                    type = copyText(definition->type);
                }
                auto *port = new PortItem(this, data.id, handle, output, handle, type, item);
                item->addPort(port);
            }
            if (join) {
                const QStringList occupied = occupiedJoinInputs_.value(data.id);
                const QString highest = handles.isEmpty() ? QString() : handles.last();
                item->setJoinInputControls(handles.size() > 2 && !occupied.contains(highest), handles.size() < 128);
            }
        }
        if (auto *subflow = dynamic_cast<SubflowItem *>(item)) {
            const int count = int(std::count_if(nodeData.cbegin(), nodeData.cend(), [&data](const NodeSnapshot &child) {
                return child.scopeId == data.id;
            }));
            subflow->setChildCount(count);
        }
    };
    for (const NodeSnapshot &data : nodeData)
        if (data.scopeId == scopeId_) addNode(data, data.position);
    for (const EdgeSnapshot &data : edgeData) {
        if (data.scopeId != visibleScopes.value(data.source) ||
            data.scopeId != visibleScopes.value(data.target)) continue;
        NodeItem *sourceNode = nodes_.value(data.source, nullptr);
        NodeItem *targetNode = nodes_.value(data.target, nullptr);
        if (!sourceNode || !targetNode) continue;
        PortItem *sourcePort = nullptr;
        PortItem *targetPort = nullptr;
        for (PortItem *port : sourceNode->ports())
            if (port->isOutput() && port->handleId() == data.sourceHandle) sourcePort = port;
        for (PortItem *port : targetNode->ports())
            if (!port->isOutput() && port->handleId() == data.targetHandle) targetPort = port;
        if (!sourcePort || !targetPort) continue;
        auto *edge = new EdgeItem(data.id, sourcePort, targetPort);
        edge->setFlag(QGraphicsItem::ItemIsSelectable, data.scopeId == scopeId_);
        edges_.insert(data.id, edge);
        edgeScopes_.insert(data.id, data.scopeId);
        addItem(edge);
        if (selectedEdges.contains(data.id)) edge->setSelected(true);
    }
    for (const QString &id : selectedNodes)
        if (nodes_.contains(id)) nodes_.value(id)->setSelected(true);
    char analysisError[512] = {};
    if (const NNInferenceReport *report = nn_app_analysis(application_, analysisError,
                                                          sizeof(analysisError))) {
        QHash<QString, QStringList> tensorLines;
        for (size_t i = 0; i < nn_inference_count(report); ++i) {
            const NNInferenceResult *result = nn_inference_at(report, i);
            if (!result || !result->node_id || result->status != NN_INFERENCE_SUCCESS) continue;
            QStringList lines;
            for (size_t j = 0; j < result->output_count; ++j) {
                const NNInferenceTensor &tensor = result->outputs[j];
                QStringList dimensions;
                for (size_t d = 0; d < tensor.dimension_count; ++d)
                    dimensions.append(copyText(tensor.dimensions[d]));
                lines.append(QStringLiteral("%1  %2  [%3]")
                    .arg(copyText(tensor.handle_id), copyText(tensor.dtype), dimensions.join(QStringLiteral(", "))));
            }
            if (lines.isEmpty() && (result->dtype || result->dimension_count)) {
                QStringList dimensions;
                for (size_t d = 0; d < result->dimension_count; ++d)
                    dimensions.append(copyText(result->dimensions[d]));
                lines.append(QStringLiteral("consumed  %1  [%2]")
                    .arg(copyText(result->dtype), dimensions.join(QStringLiteral(", "))));
            }
            tensorLines.insert(copyText(result->node_id), lines);
        }
        for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
            const QStringList lines = tensorLines.value(it.key());
            it.value()->setTensorSummary(lines.join(QLatin1Char('\n')));
        }
    }
    for (auto it = nodes_.begin(); it != nodes_.end(); ++it)
        it.value()->setToolTip(parameterDiagnostics.value(it.key()).join(QLatin1Char('\n')));
    const QRectF content = contentBounds().adjusted(-5090, -5090, 5090, 5090);
    if (content.isValid() && !content.isEmpty()) setSceneRect(content);
    refreshing_ = false;
    recomputeRoutes();
    signalBlocker.unblock();
    emit selectionChanged();
}

void GraphScene::setScope(const QString &scopeId) {
    if (scopeId_ == scopeId) return;
    cancelConnection();
    scopeId_ = scopeId;
    scheduleRefresh();
    emit scopeChanged(scopeId_);
}

void GraphScene::setFlowDirection(FlowDirection direction) {
    if (flowDirection_ == direction) return;
    flowDirection_ = direction;
    for (NodeItem *item : nodes_) item->setFlowDirection(direction);
    recomputeRoutes();
    scheduleRefresh();
    if (draftPath_ && draftSource_) updateConnection(draftSource_->scenePos());
}

void GraphScene::goToParentScope() { setScope(parentScope()); }

void GraphScene::cancelInteraction() {
    cancelConnection();
    cancelNodeDrag();
}

void GraphScene::deleteSelection() {
    QStringList edgeIds;
    QStringList nodeIds;
    for (QGraphicsItem *item : selectedItems()) {
        if (auto *edge = dynamic_cast<EdgeItem *>(item)) {
            if (edgeScopes_.value(edge->id()) == scopeId_) edgeIds.append(edge->id());
        } else if (auto *node = dynamic_cast<NodeItem *>(item)) {
            if (nodeScopes_.value(node->id()) == scopeId_) nodeIds.append(node->id());
        }
    }
    char error[512] = {};
    bool changed = false;
    if (edgeIds.isEmpty() && nodeIds.isEmpty()) return;
    if (!nn_app_begin_edit(application_, error, sizeof(error))) {
        reportError(error);
        return;
    }
    bool failed = false;
    for (const QString &id : edgeIds) {
        const QByteArray utf8 = id.toUtf8();
        if (!nn_app_disconnect(application_, utf8.constData(), error, sizeof(error))) {
            failed = true;
            break;
        }
        changed = true;
    }
    if (!failed) {
        for (const QString &id : nodeIds) {
            const QByteArray utf8 = id.toUtf8();
            if (!nn_app_remove_node(application_, utf8.constData(), error, sizeof(error))) {
                failed = true;
                break;
            }
            changed = true;
        }
    }
    char endError[512] = {};
    if (!nn_app_end_edit(application_, !failed, endError, sizeof(endError))) {
        reportError(endError);
        scheduleRefresh();
        return;
    }
    if (failed) {
        reportError(error);
        scheduleRefresh();
        return;
    }
    if (changed) {
        emit modelChanged();
        scheduleRefresh();
    }
}

QString GraphScene::selectedNodeId() const {
    for (QGraphicsItem *item : selectedItems())
        if (auto *node = dynamic_cast<NodeItem *>(item)) return node->id();
    return {};
}

NodeItem *GraphScene::nodeItem(const QString &id) const { return nodes_.value(id, nullptr); }
EdgeItem *GraphScene::edgeItem(const QString &id) const { return edges_.value(id, nullptr); }

QRectF GraphScene::contentBounds() const {
    QRectF bounds;
    for (NodeItem *node : nodes_) bounds = bounds.united(node->sceneBoundingRect());
    for (EdgeItem *edge : edges_)
        if (!edge->routePoints().isEmpty()) bounds = bounds.united(edge->sceneBoundingRect());
    return bounds;
}

void GraphScene::recomputeRoutes() {
    if (refreshing_) return;
    for (NodeItem *node : nodes_) node->setRoutingWarning(false);
    QStringList ids = edges_.keys();
    std::sort(ids.begin(), ids.end());
    QHash<QString, QVector<QVector<QPointF>>> routesByScope;
    for (const QString &id : ids) {
        EdgeItem *edge = edges_.value(id);
        PortItem *source = edge ? edge->sourcePort() : nullptr;
        PortItem *target = edge ? edge->targetPort() : nullptr;
        if (!source || !target) continue;
        const QString scope = nodeScopes_.value(source->nodeId());
        if (scope != nodeScopes_.value(target->nodeId())) {
            edge->setRoutePoints({});
            continue;
        }
        NodeItem *sourceNode = nodes_.value(source->nodeId());
        NodeItem *targetNode = nodes_.value(target->nodeId());
        if (!sourceNode || !targetNode) {
            edge->setRoutePoints({});
            continue;
        }
        const bool horizontal = flowDirection_ == FlowDirection::Horizontal;
        OrthogonalRouter::Request request;
        request.edgeId = id;
        // Preview children use uniformly scaled geometry, including routing clearance.
        request.margin = 10.0 * qMin(qAbs(sourceNode->sceneTransform().m11()),
                                     qAbs(targetNode->sceneTransform().m11()));
        request.source = {source->scenePos(), horizontal ? QPointF(-1, 0) : QPointF(0, 1),
                          sourceNode->sceneBoundingRect(), source->nodeId()};
        request.target = {target->scenePos(), horizontal ? QPointF(1, 0) : QPointF(0, -1),
                          targetNode->sceneBoundingRect(), target->nodeId()};
        for (NodeItem *obstacle : nodes_) {
            if (nodeScopes_.value(obstacle->id()) != scope) continue;
            request.obstacles.append({obstacle->sceneBoundingRect(), obstacle->id()});
        }
        request.existingRoutes = routesByScope.value(scope);
        const QVector<QPointF> route = OrthogonalRouter::route(request);
        edge->setRoutePoints(route);
        if (route.isEmpty() && sourceNode) sourceNode->setRoutingWarning(true);
        if (!route.isEmpty()) routesByScope[scope].append(route);
    }
}

QColor GraphScene::connectionDraftColor() const {
    return draftPath_ ? draftPath_->pen().color() : QColor();
}

void GraphScene::setProblemMarkers(const QHash<QString, QString> &categories) {
    problemCategories_ = categories;
    for (auto it = nodes_.cbegin(); it != nodes_.cend(); ++it)
        it.value()->setProblemCategory(categories.value(it.key()));
}

void GraphScene::revealNode(const QString &id) {
    const QByteArray idBytes = id.toUtf8();
    const NNNode *node = nn_model_find_node(nn_app_model(application_), idBytes.constData());
    if (!node) return;
    const QString scope = copyText(node->scope_id);
    if (scope != scopeId_) setScope(scope);
    QMetaObject::invokeMethod(this, [this, id] {
        refresh();
        if (NodeItem *item = nodes_.value(id, nullptr)) {
            clearSelection();
            item->setSelected(true);
            if (views().size()) views().first()->centerOn(item);
            emit selectionChanged();
        }
    }, Qt::QueuedConnection);
}

void GraphScene::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape) {
        cancelInteraction();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Delete || event->key() == Qt::Key_Backspace) {
        deleteSelection();
        event->accept();
        return;
    }
    QGraphicsScene::keyPressEvent(event);
}

void GraphScene::contextMenuEvent(QGraphicsSceneContextMenuEvent *event) {
    QGraphicsItem *target = itemAt(event->scenePos(), QTransform());
    while (target && !dynamic_cast<NodeItem *>(target) && !dynamic_cast<EdgeItem *>(target))
        target = target->parentItem();
    auto *node = dynamic_cast<NodeItem *>(target);
    auto *edge = dynamic_cast<EdgeItem *>(target);
    if (!node && !edge) {
        QGraphicsScene::contextMenuEvent(event);
        return;
    }
    clearSelection();
    target->setSelected(true);
    QMenu menu;
    QAction *remove = menu.addAction(edge ? tr("Delete connection") : tr("Delete"));
    QAction *chosen = menu.exec(event->screenPos());
    if (chosen == remove) deleteSelection();
    event->accept();
}

void GraphScene::updateEdgesForNode(const QString &nodeId) {
    Q_UNUSED(nodeId)
    if (refreshing_) return;
    recomputeRoutes();
    if (draftPath_ && draftSource_) updateConnection(draftSource_->scenePos());
}

void GraphScene::beginNodeDrag(NodeItem *item) {
    if (!item || refreshing_) return;
    dragPositions_.clear();
    for (QGraphicsItem *selected : selectedItems()) {
        if (auto *node = dynamic_cast<NodeItem *>(selected)) dragPositions_.insert(node->id(), node->pos());
    }
    if (!dragPositions_.contains(item->id())) dragPositions_.insert(item->id(), item->pos());
}

void GraphScene::commitNodeMoves() {
    if (refreshing_) return;
    for (const QString &id : lockedMovableNodes_)
        if (NodeItem *item = nodes_.value(id, nullptr)) item->setFlag(QGraphicsItem::ItemIsMovable, true);
    lockedMovableNodes_.clear();
    if (dragPositions_.isEmpty()) return;
    const auto originals = dragPositions_;
    dragPositions_.clear();
    char error[512] = {};
    if (!nn_app_begin_edit(application_, error, sizeof(error))) {
        reportError(error);
        scheduleRefresh();
        return;
    }
    bool changed = false;
    bool failed = false;
    for (auto it = originals.cbegin(); it != originals.cend(); ++it) {
        NodeItem *item = nodes_.value(it.key(), nullptr);
        if (!item || (qFuzzyCompare(item->x() + 1.0, it.value().x() + 1.0) &&
                      qFuzzyCompare(item->y() + 1.0, it.value().y() + 1.0))) continue;
        const QByteArray id = it.key().toUtf8();
        const QPointF position = item->pos();
        if (!nn_app_move_node(application_, id.constData(), position.x(), position.y(), error, sizeof(error))) {
            failed = true;
            break;
        }
        changed = true;
    }
    char endError[512] = {};
    if (!nn_app_end_edit(application_, !failed, endError, sizeof(endError))) {
        reportError(endError);
        scheduleRefresh();
        return;
    }
    if (failed) {
        reportError(error);
        scheduleRefresh();
        return;
    }
    if (changed) emit modelChanged();
    if (changed) scheduleRefresh();
}

void GraphScene::cancelNodeDrag() {
    if (dragPositions_.isEmpty()) return;
    const auto originals = dragPositions_;
    dragPositions_.clear();
    for (auto it = originals.cbegin(); it != originals.cend(); ++it) {
        if (NodeItem *item = nodes_.value(it.key(), nullptr)) {
            item->setPos(it.value());
            item->setFlag(QGraphicsItem::ItemIsMovable, false);
            lockedMovableNodes_.append(it.key());
        }
    }
}

void GraphScene::scheduleRefresh() {
    if (refreshPending_) return;
    refreshPending_ = true;
    QMetaObject::invokeMethod(this, [this] {
        refreshPending_ = false;
        refresh();
    }, Qt::QueuedConnection);
}

void GraphScene::beginConnection(PortItem *port) {
    cancelConnection();
    if (!port || !port->isOutput()) return;
    draftSource_ = port;
    const QColor color = port->outputType() == QStringLiteral("loss")
        ? QColor("#c62828") : QColor("#111111");
    draftPath_ = addPath(QPainterPath(), QPen(color, 1.8, Qt::DashLine,
                                               Qt::RoundCap, Qt::RoundJoin));
    draftPath_->setZValue(0.0);
    updateConnection(port->scenePos());
}

void GraphScene::adjustJoinInputSlots(const QString &nodeId, bool add) {
    NodeItem *item = nodes_.value(nodeId, nullptr);
    if (!item) return;
    QStringList handles;
    for (PortItem *port : item->ports())
        if (!port->isOutput()) handles.append(port->handleId());
    sortInputHandles(&handles);
    const QStringList occupied = occupiedJoinInputs_.value(nodeId);
    QStringList explicitHandles = explicitJoinInputHandles_.value(nodeId);
    if (add) {
        if (handles.size() >= 128) return;
        quint64 number = 1;
        QString candidate;
        do {
            candidate = QStringLiteral("in-%1").arg(number++);
        } while (handles.contains(candidate) || occupied.contains(candidate));
        explicitHandles.append(candidate);
    } else {
        if (handles.size() <= 2 || occupied.contains(handles.last())) return;
        const QString removed = handles.takeLast();
        if (!explicitHandles.removeOne(removed)) return;
    }
    sortInputHandles(&explicitHandles);
    explicitJoinInputHandles_[nodeId] = explicitHandles;
    scheduleRefresh();
}

void GraphScene::updateConnection(const QPointF &position) {
    if (!draftPath_ || !draftSource_) return;
    const QPointF start = draftSource_->scenePos();
    const bool horizontal = flowDirection_ == FlowDirection::Horizontal;
    const qreal primaryDelta = horizontal ? start.x() - position.x() : position.y() - start.y();
    const qreal escape = qMax<qreal>(24.0, qAbs(primaryDelta) * 0.2);
    const qreal midpoint = (horizontal ? start.x() + position.x() : start.y() + position.y()) / 2.0;
    QPainterPath path(start);
    if (horizontal) {
        path.lineTo(start.x() - escape, start.y());
        path.lineTo(midpoint, start.y());
        path.lineTo(midpoint, position.y());
    } else {
        path.lineTo(start.x(), start.y() + escape);
        path.lineTo(start.x(), midpoint);
        path.lineTo(position.x(), midpoint);
    }
    path.lineTo(position);
    draftPath_->setPath(path);
}

void GraphScene::finishConnection(PortItem *port) {
    PortItem *source = draftSource_;
    cancelConnection();
    if (!source || !port || port->isOutput()) return;
    const QByteArray sourceId = source->nodeId().toUtf8();
    const QByteArray sourceHandle = source->handleId().toUtf8();
    const QByteArray targetId = port->nodeId().toUtf8();
    const QByteArray targetHandle = port->handleId().toUtf8();
    const QByteArray edgeId = QUuid::createUuid().toString(QUuid::WithoutBraces).toUtf8();
    char error[512] = {};
    if (!nn_app_connect(application_, edgeId.constData(), sourceId.constData(), sourceHandle.constData(),
                        targetId.constData(), targetHandle.constData(), error, sizeof(error))) {
        emit errorOccurred(tr("Cannot connect nodes: %1").arg(copyText(error)));
        return;
    }
    emit modelChanged();
    scheduleRefresh();
}

void GraphScene::cancelConnection() {
    draftSource_ = nullptr;
    if (draftPath_) {
        removeItem(draftPath_);
        delete draftPath_;
        draftPath_ = nullptr;
    }
}

void GraphScene::reportError(const char *message) {
    const QString text = copyText(message);
    if (!text.isEmpty()) emit errorOccurred(text);
}

QString GraphScene::parentScope() const {
    const NNModel *model = nn_app_model(application_);
    if (!model || scopeId_.isEmpty()) return {};
    const QByteArray scope = scopeId_.toUtf8();
    const NNNode *node = nn_model_find_node(model, scope.constData());
    return node ? copyText(node->scope_id) : QString();
}
