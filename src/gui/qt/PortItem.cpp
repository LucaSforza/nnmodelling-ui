#include "PortItem.hpp"

#include "GraphScene.hpp"

#include <QGraphicsSceneHoverEvent>
#include <QGraphicsSceneMouseEvent>
#include <QPainter>

PortItem::PortItem(GraphScene *owner, QString nodeId, QString handleId, bool output,
                   QString label, QString outputType, QGraphicsItem *parent)
    : QGraphicsObject(parent), owner_(owner), nodeId_(std::move(nodeId)),
      handleId_(std::move(handleId)), label_(std::move(label)),
      outputType_(std::move(outputType)), output_(output) {
    setToolTip(output_ && !outputType_.isEmpty()
        ? QStringLiteral("%1 (%2)").arg(handleId_, outputType_) : handleId_);
    setAcceptedMouseButtons(Qt::LeftButton);
    setAcceptHoverEvents(true);
    setZValue(3.0);
}

QRectF PortItem::boundingRect() const { return QRectF(-7, -7, 14, 14); }

void PortItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) {
    if (glyphSuppressed_) return;
    painter->setRenderHint(QPainter::Antialiasing);
    const QColor typeColor = outputType_ == QStringLiteral("loss") ? QColor("#c62828")
        : outputType_ == QStringLiteral("output") ? QColor("#161616") : QColor("#ffffff");
    if (hovered_) {
        painter->setPen(QPen(QColor(42, 108, 179), 1.5));
        painter->setBrush(Qt::NoBrush);
        painter->drawEllipse(QPointF(0, 0), 6.2, 6.2);
    }
    painter->setPen(QPen(QColor(51, 65, 81), 1.2));
    painter->setBrush(typeColor);
    painter->drawEllipse(QPointF(0, 0), 4.0, 4.0);
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
