#include "NodeItem.hpp"

#include "GraphScene.hpp"
#include "PortItem.hpp"

#include <QBrush>
#include <QFont>
#include <QFontMetricsF>
#include <QGraphicsSceneMouseEvent>
#include <algorithm>
#include <QPainter>
#include <QPen>

NodeItem::NodeItem(GraphScene *owner, QString id, QString label, QString packageId,
                   const QPointF &position, const QColor &color)
    : owner_(owner), id_(std::move(id)), label_(std::move(label)),
      packageId_(std::move(packageId)), color_(color) {
    setPos(position);
    setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsGeometryChanges);
    setCacheMode(DeviceCoordinateCache);
    setZValue(1.0);
}

QRectF NodeItem::boundingRect() const { return QRectF(0.5, 0.5, width_ - 1, height_ - 1); }

void NodeItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) {
    painter->setRenderHint(QPainter::Antialiasing);
    const QRectF card(0.5, 8.5, width_ - 1.0, height_ - 17.0);
    painter->setPen(QPen(isSelected() ? QColor(43, 105, 174) : QColor(177, 188, 201),
                         isSelected() ? 2.0 : 1.0));
    painter->setBrush(color_.lighter(183));
    painter->drawRoundedRect(card, 7, 7);
    painter->setPen(Qt::NoPen);
    painter->setBrush(color_);
    painter->drawRoundedRect(QRectF(0, 0, 7, height_), 4, 4);

    painter->setPen(QColor(37, 49, 64));
    QFont title = painter->font();
    title.setBold(true);
    title.setPointSizeF(9.5);
    painter->setFont(title);
    QFontMetricsF metrics(title);
    const QString displayLabel = metrics.elidedText(label_, Qt::ElideRight, width_ - 24);
    painter->drawText(QRectF(14, 18, width_ - 23, 19), Qt::AlignLeft | Qt::AlignVCenter,
                      displayLabel);
    QFont detail = painter->font();
    detail.setBold(false);
    detail.setPointSizeF(7.2);
    painter->setFont(detail);
    painter->setPen(QColor(112, 125, 141));
    painter->drawText(QRectF(14, 39, width_ - 23, 18), Qt::AlignLeft | Qt::AlignVCenter,
                      QFontMetricsF(detail).elidedText(packageId_, Qt::ElideMiddle, width_ - 23));
    painter->setPen(QColor(94, 107, 122));
    QFont portFont = painter->font();
    portFont.setPointSizeF(6.8);
    painter->setFont(portFont);
    for (PortItem *port : ports_) {
        const qreal y = port->isOutput() ? height_ - 21.0 : 0.0;
        painter->drawText(QRectF(port->x() - 46.0, y, 92.0, 10.0),
                          Qt::AlignHCenter | Qt::AlignVCenter, port->label());
    }
}

void NodeItem::addPort(PortItem *port) {
    ports_.append(port);
    for (bool output : {true, false}) {
        int order = 0;
        const int count = std::count_if(ports_.cbegin(), ports_.cend(), [output](PortItem *p) {
            return p->isOutput() == output;
        });
        for (PortItem *candidate : ports_) {
            if (candidate->isOutput() != output) continue;
            const qreal x = width_ * (order + 1.0) / (count + 1.0);
            candidate->setPos(x, output ? height_ - 9.0 : 9.0);
            ++order;
        }
    }
}

void NodeItem::setChildCount(int) {}

QVariant NodeItem::itemChange(GraphicsItemChange change, const QVariant &value) {
    if (change == ItemPositionHasChanged && owner_)
        owner_->updateEdgesForNode(id_);
    return QGraphicsObject::itemChange(change, value);
}

void NodeItem::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    QGraphicsObject::mousePressEvent(event);
    if (owner_) owner_->beginNodeDrag(this);
}

void NodeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    QGraphicsObject::mouseReleaseEvent(event);
    if (owner_) owner_->commitNodeMoves();
}
