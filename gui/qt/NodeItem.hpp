#ifndef NN_NODE_ITEM_HPP
#define NN_NODE_ITEM_HPP

#include <QGraphicsObject>
#include <QColor>
#include <QList>
#include <QString>

class GraphScene;
class PortItem;

class NodeItem : public QGraphicsObject {
    Q_OBJECT
public:
    NodeItem(GraphScene *owner, QString id, QString label, QString packageId,
             const QPointF &position, const QColor &color);
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;
    QString id() const { return id_; }
    void addPort(PortItem *port);
    QList<PortItem *> ports() const { return ports_; }
    virtual void setChildCount(int count);

protected:
    QVariant itemChange(GraphicsItemChange change, const QVariant &value) override;
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;
    GraphScene *owner_;
    QString id_;
    QString label_;
    QString packageId_;
    QColor color_;
    QList<PortItem *> ports_;
    qreal width_ = 190.0;
    qreal height_ = 96.0;
};

#endif
