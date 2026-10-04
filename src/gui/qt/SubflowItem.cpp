#include "SubflowItem.hpp"

#include "GraphScene.hpp"
#include "PortItem.hpp"

#include <QGraphicsSceneMouseEvent>
#include <QPainter>
#include <utility>
#include <algorithm>

SubflowItem::SubflowItem(GraphScene *owner, QString id, QString label, QString packageId,
                         const QPointF &position, const QColor &color,
                         QList<QPair<QString, QString>> topParameters,
                         QList<QPair<QString, QString>> bottomParameters)
    : NodeItem(owner, std::move(id), std::move(label), std::move(packageId), position, color,
               std::move(topParameters), std::move(bottomParameters)) {
    height_ += 40.0;
}

void SubflowItem::setChildCount(int count) { childCount_ = count; }

QRectF SubflowItem::boundingRect() const {
    return NodeItem::boundingRect().united(expandedPreviewRect_);
}

void SubflowItem::setFlowDirection(FlowDirection direction) {
    NodeItem::setFlowDirection(direction);
    if (!expandedContentRect_.isEmpty()) setExpandedPreviewRect(expandedContentRect_);
}

void SubflowItem::setExpandedPreviewRect(const QRectF &rect) {
    const QRectF expanded = rect.isEmpty() ? QRectF() :
        rect.united(NodeItem::boundingRect()).adjusted(-30, -30, 30, 30);
    if (expandedPreviewRect_ != expanded) prepareGeometryChange();
    expandedContentRect_ = rect;
    expandedPreviewRect_ = expanded;
    setExternalPortsOnBoundary(!expanded.isEmpty());
    if (expanded.isEmpty()) layoutPorts();
    else for (bool output : {true, false}) {
        int count = 0;
        const int total = std::count_if(ports_.cbegin(), ports_.cend(), [output](PortItem *port) {
            return port->isOutput() == output;
        });
        for (PortItem *port : ports_) {
            if (port->isOutput() != output) continue;
            const qreal fraction = (count++ + 1.0) / (total + 1.0);
            if (flowDirection_ == FlowDirection::Vertical)
                port->setPos(expanded.left() + fraction * expanded.width(), output ? expanded.bottom() : expanded.top());
            else
                port->setPos(output ? expanded.left() : expanded.right(), expanded.top() + fraction * expanded.height());
        }
    }
    update();
}

QRectF SubflowItem::footerRect() const {
    const qreal y = height_ - 30.0 - bottomParameters_.size() * 25.0;
    return QRectF(10, y, width_ - 19, 21);
}

void SubflowItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
    if (!expandedPreviewRect_.isEmpty()) {
        painter->setPen(QPen(QColor(128, 157, 183), 1.5, Qt::DashLine));
        painter->setBrush(QColor(246, 249, 252, 100));
        painter->drawRoundedRect(expandedPreviewRect_, 12, 12);
    }
    NodeItem::paint(painter, option, widget);
    QFont footerFont = painter->font();
    footerFont.setPointSizeF(7.2);
    painter->setFont(footerFont);
    painter->setPen(QPen(QColor(118, 151, 183), 1.0, Qt::DashLine));
    painter->setBrush(QColor(240, 246, 252));
    const qreal footerY = footerRect().top();
    painter->drawRoundedRect(footerRect(), 5, 5);
    painter->setPen(QColor(35, 57, 81));
    painter->setFont(footerFont);
    painter->drawText(QRectF(14, footerY + 2, width_ - 23, 16), Qt::AlignLeft | Qt::AlignVCenter,
                      QStringLiteral("Subflow  ·  %1 nodes  %2")
                          .arg(childCount_).arg(expandedPreviewRect_.isEmpty() ? QStringLiteral("›")
                                                                               : QStringLiteral("▾")));
}

void SubflowItem::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::LeftButton && footerRect().contains(event->pos())) {
        owner_->toggleExpanded(id_);
        event->accept();
        return;
    }
    NodeItem::mousePressEvent(event);
}

void SubflowItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        owner_->setScope(id_);
        event->accept();
        return;
    }
    NodeItem::mouseDoubleClickEvent(event);
}
