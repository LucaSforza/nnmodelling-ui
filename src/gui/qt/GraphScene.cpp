#include "GraphScene.hpp"

#include "EdgeItem.hpp"
#include "NodeItem.hpp"
#include "PortItem.hpp"
#include "SubflowItem.hpp"

#include "application/application.h"
#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"

#include <QGraphicsPathItem>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QPainterPath>
#include <QPainter>
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
        joinInputHandles_.clear();
        suppressedJoinInputs_.clear();
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
            if (!edge || copyText(edge->scope_id) != scopeId_) continue;
            edgeData.push_back({copyText(edge->id), copyText(edge->source_id),
                                copyText(edge->source_handle_id), copyText(edge->target_id),
                                copyText(edge->target_handle_id)});
            occupiedJoinInputs_[copyText(edge->target_id)].append(copyText(edge->target_handle_id));
        }
    }

    clear();
    nodes_.clear();
    edges_.clear();
    draftSource_ = nullptr;
    draftPath_ = nullptr;
    for (const NodeSnapshot &data : nodeData) {
        if (data.scopeId != scopeId_) continue;
        const NNNode *node = nn_model_find_node(nn_app_model(application_),
                                                data.id.toUtf8().constData());
        const QColor color = packageColor(application_, data.packageId,
                                          node ? copyText(node->package_version) : QString());
        QList<QPair<QString, QString>> topParameters;
        QList<QPair<QString, QString>> bottomParameters;
        const NNProject *project = nn_app_project(application_);
        const NNCatalog *catalog = project ? nn_project_catalog(project) : nullptr;
        const QByteArray packageId = data.packageId.toUtf8();
        const QByteArray packageVersion = node ? QByteArray(node->package_version) : QByteArray();
        const NNPackage *package = catalog
            ? nn_catalog_find(catalog, packageId.constData(), packageVersion.constData()) : nullptr;
        for (size_t i = 0; package && i < package->parameter_count; ++i) {
            const NNParameterDef &definition = package->parameters[i];
            if (!definition.position || !definition.key) continue;
            const QByteArray nodeId = data.id.toUtf8();
            char *raw = nn_app_parameter_text(application_, nodeId.constData(), definition.key);
            const QString value = copyText(raw);
            nn_app_free_text(raw);
            auto &rows = std::strcmp(definition.position, "top") == 0
                ? topParameters : bottomParameters;
            rows.append(qMakePair(QString::fromUtf8(definition.key), value));
        }
        NodeItem *item = data.subflow
            ? static_cast<NodeItem *>(new SubflowItem(this, data.id, data.label, data.packageId,
                                                       data.position, color, topParameters, bottomParameters))
            : new NodeItem(this, data.id, data.label, data.packageId, data.position, color,
                            topParameters, bottomParameters);
        if (data.kind == QStringLiteral("input") || data.kind == QStringLiteral("output") ||
            data.kind == QStringLiteral("loss-output")) item->setBoundaryKind(data.kind);
        const bool join = data.kind == QStringLiteral("join");
        item->setJoinNode(join);
        item->setFlowDirection(flowDirection_);
        nodes_.insert(data.id, item);
        item->setProblemCategory(problemCategories_.value(data.id));
        addItem(item);
        for (bool output : {true, false}) {
            const QByteArray nodeId = data.id.toUtf8();
            QStringList handles;
            if (output) {
                for (size_t index = 0; package && index < package->output_count; ++index)
                    if (package->outputs[index].id)
                        handles.append(QString::fromUtf8(package->outputs[index].id));
            } else if (join) {
                const QStringList occupied = occupiedJoinInputs_.value(data.id);
                const QStringList suppressed = suppressedJoinInputs_.value(data.id);
                auto appendIfVisible = [&handles, &occupied, &suppressed](const QString &handle) {
                    if ((occupied.contains(handle) || !suppressed.contains(handle)) &&
                        !handles.contains(handle)) handles.append(handle);
                };
                for (const QString &handle : joinInputHandles_.value(data.id))
                    appendIfVisible(handle);
                for (const QString &handle : occupied) appendIfVisible(handle);
                const size_t count = nn_app_port_count(application_, nodeId.constData(), false);
                for (size_t index = 0; index < count; ++index) {
                    char handle[128] = {};
                    if (nn_app_port_id(application_, nodeId.constData(), false, index,
                                       handle, sizeof(handle))) appendIfVisible(QString::fromUtf8(handle));
                }
                for (int number = 1; handles.size() < 2; ++number)
                    appendIfVisible(QStringLiteral("in-%1").arg(number));
                sortInputHandles(&handles);
                joinInputHandles_[data.id] = handles;
            } else {
                const size_t count = nn_app_port_count(application_, nodeId.constData(), false);
                for (size_t index = 0; index < count; ++index) {
                    char handle[128] = {};
                    if (nn_app_port_id(application_, nodeId.constData(), false, index,
                                       handle, sizeof(handle))) handles.append(QString::fromUtf8(handle));
                }
            }
            for (int index = 0; index < handles.size(); ++index) {
                QString handleId;
                QString type;
                if (output) {
                    handleId = handles[index];
                    const NNOutputDef *definition = nullptr;
                    for (size_t outputIndex = 0; package && outputIndex < package->output_count; ++outputIndex)
                        if (package->outputs[outputIndex].id &&
                            handleId == QString::fromUtf8(package->outputs[outputIndex].id))
                            definition = &package->outputs[outputIndex];
                    if (!definition) continue;
                    type = copyText(definition->type);
                } else {
                    handleId = handles[index];
                }
                auto *port = new PortItem(this, data.id, handleId, output, handleId, type, item);
                item->addPort(port);
            }
            if (join) {
                const QStringList occupied = occupiedJoinInputs_.value(data.id);
                const QString highest = handles.isEmpty() ? QString() : handles.last();
                const bool canRemove = handles.size() > 2 && !occupied.contains(highest);
                item->setJoinInputControls(canRemove, handles.size() < 128);
            }
        }
    }
    for (const NodeSnapshot &data : nodeData) {
        auto *subflow = dynamic_cast<SubflowItem *>(nodes_.value(data.id));
        if (!subflow) continue;
        const int count = int(std::count_if(nodeData.cbegin(), nodeData.cend(), [&data](const NodeSnapshot &child) {
            return child.scopeId == data.id;
        }));
        subflow->setChildCount(count);
    }
    for (const EdgeSnapshot &data : edgeData) {
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
        edges_.insert(data.id, edge);
        addItem(edge);
        if (selectedEdges.contains(data.id)) edge->setSelected(true);
    }
    for (const QString &id : selectedNodes)
        if (nodes_.contains(id)) nodes_.value(id)->setSelected(true);
    const QRectF content = itemsBoundingRect().adjusted(-5090, -5090, 5090, 5090);
    if (content.isValid() && !content.isEmpty()) setSceneRect(content);
    refreshing_ = false;
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
    for (EdgeItem *edge : edges_) edge->updatePath();
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
        if (auto *edge = dynamic_cast<EdgeItem *>(item)) edgeIds.append(edge->id());
        else if (auto *node = dynamic_cast<NodeItem *>(item)) nodeIds.append(node->id());
    }
    char error[512] = {};
    bool changed = false;
    for (const QString &id : edgeIds) {
        const QByteArray utf8 = id.toUtf8();
        if (!nn_app_disconnect(application_, utf8.constData(), error, sizeof(error))) {
            reportError(error);
            break;
        }
        changed = true;
    }
    if (std::strlen(error) == 0) {
        for (const QString &id : nodeIds) {
            const QByteArray utf8 = id.toUtf8();
            if (!nn_app_remove_node(application_, utf8.constData(), error, sizeof(error))) {
                reportError(error);
                break;
            }
            changed = true;
        }
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

void GraphScene::updateEdgesForNode(const QString &nodeId) {
    Q_UNUSED(nodeId)
    for (EdgeItem *edge : edges_) edge->updatePath();
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
    bool changed = false;
    char error[512] = {};
    for (auto it = originals.cbegin(); it != originals.cend(); ++it) {
        NodeItem *item = nodes_.value(it.key(), nullptr);
        if (!item || (qFuzzyCompare(item->x() + 1.0, it.value().x() + 1.0) &&
                      qFuzzyCompare(item->y() + 1.0, it.value().y() + 1.0))) continue;
        const QByteArray id = it.key().toUtf8();
        if (!nn_app_move_node(application_, id.constData(), item->x(), item->y(), error, sizeof(error))) {
            reportError(error);
            break;
        }
        changed = true;
    }
    if (changed) emit modelChanged();
    if (changed || std::strlen(error) != 0) scheduleRefresh();
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
    QStringList handles = joinInputHandles_.value(nodeId);
    if (handles.isEmpty()) return;
    const QStringList occupied = occupiedJoinInputs_.value(nodeId);
    if (add) {
        if (handles.size() >= 128) return;
        quint64 number = 1;
        QString candidate;
        do {
            candidate = QStringLiteral("in-%1").arg(number++);
        } while (handles.contains(candidate) || occupied.contains(candidate));
        handles.append(candidate);
        suppressedJoinInputs_[nodeId].removeAll(candidate);
    } else {
        if (handles.size() <= 2 || occupied.contains(handles.last())) return;
        const QString removed = handles.takeLast();
        suppressedJoinInputs_[nodeId].append(removed);
    }
    sortInputHandles(&handles);
    joinInputHandles_[nodeId] = handles;
    scheduleRefresh();
}

void GraphScene::updateConnection(const QPointF &position) {
    if (!draftPath_ || !draftSource_) return;
    const QPointF start = draftSource_->scenePos();
    const bool horizontal = flowDirection_ == FlowDirection::Horizontal;
    const qreal delta = horizontal ? start.x() - position.x() : position.y() - start.y();
    const qreal bend = qMax<qreal>(32.0, qAbs(delta) * 0.4);
    QPainterPath path(start);
    const QPointF tangent = horizontal ? QPointF(-bend, 0) : QPointF(0, bend);
    path.cubicTo(start + tangent, position - tangent, position);
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
