#include "EdgeItem.hpp"

#include "GraphScene.hpp"
#include "PortItem.hpp"

#include <QPainter>
#include <QPainterPath>
#include <QPen>
#include <cmath>

EdgeItem::EdgeItem(QString id, PortItem *source, PortItem *target)
    : id_(std::move(id)), source_(source), target_(target) {
    const QColor color = source_ && source_->outputType() == QStringLiteral("loss")
        ? QColor("#c62828") : QColor("#111111");
    setPen(QPen(color, 1.9, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    setFlag(ItemIsSelectable);
    setZValue(-1.0);
}

void EdgeItem::updatePath() {
    if (auto *owner = dynamic_cast<GraphScene *>(scene())) owner->recomputeRoutes();
}

void EdgeItem::setRoutePoints(const QVector<QPointF> &points) {
    prepareGeometryChange();
    routePoints_ = points;
    QPainterPath polyline;
    if (!routePoints_.isEmpty()) {
        polyline.moveTo(routePoints_.first());
        for (int i = 1; i < routePoints_.size(); ++i) polyline.lineTo(routePoints_[i]);
    }
    setPath(polyline);
    update();
}

QRectF EdgeItem::boundingRect() const {
    return QGraphicsPathItem::boundingRect().adjusted(-8, -8, 8, 8);
}

void EdgeItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    QGraphicsPathItem::paint(painter, option, widget);
    if (routePoints_.size() < 2) return;
    const QPointF end = routePoints_.last();
    const QPointF before = routePoints_[routePoints_.size() - 2];
    const QPointF delta = end - before;
    const qreal length = std::hypot(delta.x(), delta.y());
    if (length < 1e-7) return;
    const QPointF direction = delta / length;
    const QPointF normal(-direction.y(), direction.x());
    constexpr qreal arrowLength = 8.0;
    constexpr qreal arrowHalfWidth = 4.0;
    QPolygonF arrow;
    arrow << end << end - direction * arrowLength + normal * arrowHalfWidth
          << end - direction * arrowLength - normal * arrowHalfWidth;
    painter->save();
    painter->setPen(Qt::NoPen);
    painter->setBrush(pen().color());
    painter->drawPolygon(arrow);
    painter->restore();
}
