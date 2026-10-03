#ifndef NN_GRAPH_SCENE_HPP
#define NN_GRAPH_SCENE_HPP

#include <QGraphicsScene>
#include <QHash>
#include <QColor>
#include <QPointF>
#include <QString>
#include <QStringList>

#include "application/application.h"
#include "NodeItem.hpp"

class EdgeItem;
class PortItem;
class QGraphicsPathItem;
class QPainter;

class GraphScene final : public QGraphicsScene {
    Q_OBJECT
public:
    explicit GraphScene(NNApplication *application, QObject *parent = nullptr);

    void refresh();
    void setScope(const QString &scopeId);
    void setFlowDirection(FlowDirection direction);
    FlowDirection flowDirection() const { return flowDirection_; }
    QString scope() const { return scopeId_; }
    void deleteSelection();
    void goToParentScope();
    void cancelInteraction();
    QString selectedNodeId() const;
    NodeItem *nodeItem(const QString &id) const;
    EdgeItem *edgeItem(const QString &id) const;
    void setProblemMarkers(const QHash<QString, QString> &categories);
    void revealNode(const QString &id);
    QColor connectionDraftColor() const;

signals:
    void modelChanged();
    void scopeChanged(const QString &scopeId);
    void errorOccurred(const QString &message);

protected:
    void keyPressEvent(QKeyEvent *event) override;
    void drawBackground(QPainter *painter, const QRectF &rect) override;

private:
    friend class NodeItem;
    friend class PortItem;
    void updateEdgesForNode(const QString &nodeId);
    void beginNodeDrag(NodeItem *item);
    void commitNodeMoves();
    void cancelNodeDrag();
    void scheduleRefresh();
    void beginConnection(PortItem *port);
    void adjustJoinInputSlots(const QString &nodeId, bool add);
    void updateConnection(const QPointF &position);
    void finishConnection(PortItem *port);
    void cancelConnection();
    void reportError(const char *message);
    QString parentScope() const;

    NNApplication *application_;
    const void *projectIdentity_ = nullptr;
    FlowDirection flowDirection_ = FlowDirection::Vertical;
    QString scopeId_;
    QHash<QString, NodeItem *> nodes_;
    QHash<QString, EdgeItem *> edges_;
    QHash<QString, QString> problemCategories_;
    QHash<QString, QStringList> joinInputHandles_;
    QHash<QString, QStringList> suppressedJoinInputs_;
    QHash<QString, QStringList> occupiedJoinInputs_;
    PortItem *draftSource_ = nullptr;
    QGraphicsPathItem *draftPath_ = nullptr;
    bool refreshing_ = false;
    bool refreshPending_ = false;
    QHash<QString, QPointF> dragPositions_;
    QStringList lockedMovableNodes_;
};

#endif
