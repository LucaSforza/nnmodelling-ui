#include "SubflowItem.hpp"

#include "GraphScene.hpp"
#include "PortItem.hpp"

#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <utility>

SubflowItem::SubflowItem(GraphScene *owner, QString id, QString label, QString packageId,
                         const QPointF &position, const QColor &color,
                         QList<QPair<QString, QString>> topParameters,
                         QList<QPair<QString, QString>> bottomParameters)
    : NodeItem(owner, std::move(id), std::move(label), std::move(packageId), position, color,
               std::move(topParameters), std::move(bottomParameters)) {
    height_ += 40.0;
}

void SubflowItem::setChildCount(int count) { childCount_ = count; }

QRectF SubflowItem::boundingRect() const { return NodeItem::boundingRect(); }

void SubflowItem::setFlowDirection(FlowDirection direction) { NodeItem::setFlowDirection(direction); }

QRectF SubflowItem::footerRect() const {
    const qreal y = height_ - 33.0 - bottomParameters_.size() * 25.0;
    return QRectF(10, y, width_ - 20, 26);
}

void SubflowItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    NodeItem::paint(painter, option, widget);
    QFont footerFont = painter->font();
    footerFont.setBold(true);
    footerFont.setPointSizeF(8.5);
    painter->setFont(footerFont);
    painter->setPen(QPen(QColor(42, 92, 139), 1.0));
    painter->setBrush(QColor(222, 237, 250));
    painter->drawRoundedRect(footerRect(), 5, 5);
    painter->setPen(QColor(27, 67, 105));
    painter->drawText(footerRect(), Qt::AlignCenter,
        QStringLiteral("Enter subflow  ·  %1 nodes  ›").arg(childCount_));
}

void SubflowItem::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::LeftButton && footerRect().contains(event->pos())) {
        owner_->setScope(id_);
        event->accept();
        return;
    }
    NodeItem::mousePressEvent(event);
}
