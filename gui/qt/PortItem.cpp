#include "PortItem.hpp"

#include "GraphScene.hpp"

#include <QGraphicsSceneHoverEvent>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>

PortItem::PortItem(GraphScene *owner, QString nodeId, QString handleId, bool output,
                   QString label, QGraphicsItem *parent)
    : QGraphicsObject(parent), owner_(owner), nodeId_(std::move(nodeId)),
      handleId_(std::move(handleId)), label_(std::move(label)), output_(output) {
    setAcceptedMouseButtons(Qt::LeftButton);
    setAcceptHoverEvents(true);
    setZValue(3.0);
}

QRectF PortItem::boundingRect() const { return QRectF(-7, -7, 14, 14); }

void PortItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) {
    painter->setRenderHint(QPainter::Antialiasing);
    painter->setPen(QPen(hovered_ ? QColor(42, 108, 179) : QColor(51, 65, 81), 1.2));
    painter->setBrush(hovered_ ? QColor(213, 232, 250) : QColor(255, 255, 255));
    painter->drawEllipse(QPointF(0, 0), hovered_ ? 4.7 : 4.0, hovered_ ? 4.7 : 4.0);
}

QPainterPath PortItem::shape() const {
    QPainterPath path;
    path.addEllipse(QPointF(0, 0), 7, 7);
    return path;
}

void PortItem::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        owner_->beginConnection(this);
        event->accept();
        return;
    }
    QGraphicsObject::mousePressEvent(event);
}

void PortItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
    owner_->updateConnection(event->scenePos());
    event->accept();
}

void PortItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    QGraphicsItem *hit = scene()->itemAt(event->scenePos(), QTransform());
    auto *port = dynamic_cast<PortItem *>(hit);
    owner_->finishConnection(port);
    event->accept();
}

void PortItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
    hovered_ = true;
    update();
    QGraphicsObject::hoverEnterEvent(event);
}

void PortItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
    hovered_ = false;
    update();
    QGraphicsObject::hoverLeaveEvent(event);
}
