#include "SubflowItem.hpp"

#include "GraphScene.hpp"

#include <QGraphicsSceneMouseEvent>
#include <QPainter>

void SubflowItem::setChildCount(int count) { childCount_ = count; }

void SubflowItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    NodeItem::paint(painter, option, widget);
    QFont footerFont = painter->font();
    footerFont.setPointSizeF(7.2);
    painter->setFont(footerFont);
    painter->setPen(QPen(QColor(118, 151, 183), 1.0, Qt::DashLine));
    painter->setBrush(QColor(240, 246, 252));
    painter->drawRoundedRect(QRectF(10, 53, width_ - 19, 20), 4, 4);
    painter->setPen(QColor(67, 103, 143));
    painter->setFont(footerFont);
    painter->drawText(QRectF(14, 55, width_ - 23, 15), Qt::AlignLeft | Qt::AlignVCenter,
                      QStringLiteral("Subflow  ·  %1 nodes  ›").arg(childCount_));
}

void SubflowItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        owner_->setScope(id_);
        event->accept();
        return;
    }
    NodeItem::mouseDoubleClickEvent(event);
}
