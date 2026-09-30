#ifndef NN_NODE_ITEM_HPP
#define NN_NODE_ITEM_HPP

#include <QGraphicsObject>
#include <QColor>
#include <QList>
#include <QString>
#include <QPair>

class GraphScene;
class PortItem;

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
    QList<QPair<QString, QString>> topParameters_;
    QList<QPair<QString, QString>> bottomParameters_;
    qreal width_ = 190.0;
    qreal height_ = 96.0;
};

#endif
