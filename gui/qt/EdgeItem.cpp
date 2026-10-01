#include "EdgeItem.hpp"

#include "PortItem.hpp"

#include <QPainterPath>
#include <QPen>

EdgeItem::EdgeItem(QString id, PortItem *source, PortItem *target)
    : id_(std::move(id)), source_(source), target_(target) {
    const QColor color = source_ && source_->outputType() == QStringLiteral("loss")
        ? QColor("#c62828") : QColor("#111111");
    setPen(QPen(color, 1.9, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    setFlag(ItemIsSelectable);
    setZValue(-1.0);
    updatePath();
}

void EdgeItem::updatePath() {
    if (!source_ || !target_) return;
    const QPointF start = source_->scenePos();
    const QPointF end = target_->scenePos();
    const qreal bend = qMax<qreal>(36.0, qAbs(end.y() - start.y()) * 0.48);
    QPainterPath curve(start);
    curve.cubicTo(start + QPointF(0, bend), end - QPointF(0, bend), end);
    setPath(curve);
    QPolygonF arrow;
    const QPointF tangent = end - curve.pointAtPercent(0.985);
    const qreal angle = std::atan2(tangent.y(), tangent.x());
    const qreal size = 7.5;
    arrow << end
          << end - QPointF(std::cos(angle - 0.48) * size, std::sin(angle - 0.48) * size)
          << end - QPointF(std::cos(angle + 0.48) * size, std::sin(angle + 0.48) * size);
    QPainterPath full = curve;
    full.addPolygon(arrow);
    setPath(full);
}
