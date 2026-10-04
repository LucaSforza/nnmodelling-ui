#ifndef NN_EDGE_ITEM_HPP
#define NN_EDGE_ITEM_HPP

#include <QGraphicsPathItem>
#include <QPointF>
#include <QString>
#include <QVector>

class PortItem;

class EdgeItem final : public QGraphicsPathItem {
public:
    EdgeItem(QString id, PortItem *source, PortItem *target);
    QString id() const { return id_; }
    PortItem *sourcePort() const { return source_; }
    PortItem *targetPort() const { return target_; }
    QVector<QPointF> routePoints() const { return routePoints_; }
    bool routingFailed() const { return routePoints_.isEmpty(); }
    void setRoutePoints(const QVector<QPointF> &points);
    void updatePath();
    QRectF boundingRect() const override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;
private:
    QString id_;
    PortItem *source_;
    PortItem *target_;
    QVector<QPointF> routePoints_;
};

#endif
