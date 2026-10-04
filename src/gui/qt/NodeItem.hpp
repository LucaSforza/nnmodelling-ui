#ifndef NN_NODE_ITEM_HPP
#define NN_NODE_ITEM_HPP

#include <QGraphicsObject>
#include <QColor>
#include <QList>
#include <QString>
#include <QPair>

class GraphScene;
class PortItem;
class QGraphicsRectItem;
class QGraphicsSimpleTextItem;

enum class FlowDirection { Vertical, Horizontal };

class NodeItem : public QGraphicsObject {
    Q_OBJECT
public:
    NodeItem(GraphScene *owner, QString id, QString label, QString packageId,
             const QPointF &position, const QColor &color,
             QList<QPair<QString, QString>> topParameters = {},
             QList<QPair<QString, QString>> bottomParameters = {});
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;
    QString id() const { return id_; }
    void addPort(PortItem *port);
    QList<PortItem *> ports() const { return ports_; }
    QList<QPair<QString, QString>> topParameters() const { return topParameters_; }
    QList<QPair<QString, QString>> bottomParameters() const { return bottomParameters_; }
    virtual void setChildCount(int count);
    void setProblemCategory(const QString &category);
    void setBoundaryKind(const QString &kind);
    virtual void setFlowDirection(FlowDirection direction);
    void setJoinNode(bool join);
    void setJoinInputControls(bool canRemove, bool canAdd);
    QRectF joinRemoveControlRect() const { return joinRemoveRect_; }
    QRectF joinAddControlRect() const { return joinAddRect_; }
    QRectF junctionRect() const { return junctionRect_; }
    QString boundaryKind() const { return boundaryKind_; }
    QString problemCategory() const { return problemCategory_; }
    void setReadOnlyPreview(bool preview);
    bool isReadOnlyPreview() const { return readOnlyPreview_; }
    void setTensorSummary(const QString &summary);
    QString tensorSummary() const { return tensorSummary_; }
    void setTensorPopupVisible(bool visible);
    void setRoutingWarning(bool warning);
    void setExternalPortsOnBoundary(bool enabled) { externalPortsOnBoundary_ = enabled; }

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    void hoverEnterEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent *event) override;
    void hoverLeaveEvent(QGraphicsSceneHoverEvent *event) override;
    virtual QRectF tensorHoverRegion() const { return NodeItem::boundingRect(); }
    GraphScene *owner_;
    QString id_;
    QString label_;
    QString packageId_;
    QColor color_;
    QString problemCategory_;
    QString boundaryKind_;
    bool joinNode_ = false;
    bool readOnlyPreview_ = false;
    bool externalPortsOnBoundary_ = false;
    QString tensorSummary_;
    QGraphicsRectItem *tensorPopupBackground_ = nullptr;
    QGraphicsSimpleTextItem *tensorPopupText_ = nullptr;
    QGraphicsSimpleTextItem *routingWarningItem_ = nullptr;
    bool canRemoveJoinInput_ = false;
    bool canAddJoinInput_ = false;
    FlowDirection flowDirection_ = FlowDirection::Vertical;
    QList<PortItem *> ports_;
    QList<QPair<QString, QString>> topParameters_;
    QList<QPair<QString, QString>> bottomParameters_;
    qreal width_ = 190.0;
    qreal contentWidth_ = 190.0;
    qreal height_ = 96.0;
    QRectF joinRemoveRect_;
    QRectF joinAddRect_;
    QRectF junctionRect_;
    void layoutPorts();
    void layoutJoin();
};

#endif
