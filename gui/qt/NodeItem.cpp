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
#include <QStyleOptionGraphicsItem>

NodeItem::NodeItem(GraphScene *owner, QString id, QString label, QString packageId,
                   const QPointF &position, const QColor &color,
                   QList<QPair<QString, QString>> topParameters,
                   QList<QPair<QString, QString>> bottomParameters)
    : owner_(owner), id_(std::move(id)), label_(std::move(label)),
      packageId_(std::move(packageId)), color_(color),
      topParameters_(std::move(topParameters)), bottomParameters_(std::move(bottomParameters)) {
    const int rows = topParameters_.size() + bottomParameters_.size();
    height_ = 96.0 + rows * 25.0;
    qreal required = 190.0;
    QFont rowFont;
    rowFont.setPointSizeF(8.5);
    QFontMetricsF metrics(rowFont);
    QFont valueFont = rowFont;
    valueFont.setBold(true);
    const QFontMetricsF valueMetrics(valueFont);
    for (const auto &entry : topParameters_ + bottomParameters_)
        required = qMax(required, metrics.horizontalAdvance(entry.first) +
                         valueMetrics.horizontalAdvance(entry.second) + 64.0);
    width_ = required;
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
    const QColor tinted(
        qRound(color_.red() * 0.20 + 255.0 * 0.80),
        qRound(color_.green() * 0.20 + 255.0 * 0.80),
        qRound(color_.blue() * 0.20 + 255.0 * 0.80));
    painter->setBrush(tinted);
    painter->drawRoundedRect(card, 7, 7);
    painter->setPen(Qt::NoPen);
    painter->setBrush(color_);
    painter->drawRoundedRect(QRectF(0, 0, 7, height_), 4, 4);

    painter->setPen(QColor(37, 49, 64));
    QFont title = painter->font();
    title.setBold(true);
    title.setPointSizeF(11.0);
    painter->setFont(title);
    QFontMetricsF metrics(title);
    const QString displayLabel = metrics.elidedText(label_, Qt::ElideRight, width_ - 24);
    const qreal titleY = 18.0 + topParameters_.size() * 25.0;
    painter->drawText(QRectF(14, titleY, width_ - 23, 19), Qt::AlignLeft | Qt::AlignVCenter,
                      displayLabel);
    QFont detail = painter->font();
    detail.setBold(false);
    detail.setPointSizeF(7.2);
    painter->setFont(detail);
    painter->setPen(QColor(112, 125, 141));
    painter->drawText(QRectF(14, titleY + 20, width_ - 23, 18), Qt::AlignLeft | Qt::AlignVCenter,
                      QFontMetricsF(detail).elidedText(packageId_, Qt::ElideMiddle, width_ - 23));
    auto drawRows = [painter, this](const QList<QPair<QString, QString>> &rows, qreal firstY) {
        QFont keyFont = painter->font();
        keyFont.setBold(false);
        keyFont.setPointSizeF(8.5);
        QFont valueFont = keyFont;
        valueFont.setBold(true);
        painter->setFont(keyFont);
        for (int i = 0; i < rows.size(); ++i) {
            const qreal y = firstY + i * 25.0;
            const QRectF row(12, y, width_ - 24, 22);
            const auto &entry = rows[i];
            const qreal badgeWidth = qMin(row.width() - 42.0,
                QFontMetricsF(valueFont).horizontalAdvance(entry.second) + 18.0);
            const QRectF badge(row.right() - badgeWidth, row.top(), badgeWidth, row.height());
            painter->setPen(QColor(49, 64, 81));
            painter->drawText(QRectF(row.left() + 5, row.top(),
                                     row.width() - badgeWidth - 12, row.height()),
                              Qt::AlignLeft | Qt::AlignVCenter,
                              QFontMetricsF(keyFont).elidedText(entry.first, Qt::ElideRight,
                                  row.width() - badgeWidth - 17));
            painter->setPen(QPen(QColor(148, 164, 181), 0.8));
            painter->setBrush(QColor(255, 255, 255));
            painter->drawRoundedRect(badge, 4, 4);
            painter->setPen(QColor(35, 48, 64));
            painter->setFont(valueFont);
            painter->drawText(badge.adjusted(5, 0, -5, 0), Qt::AlignRight | Qt::AlignVCenter,
                QFontMetricsF(valueFont).elidedText(entry.second, Qt::ElideLeft, badge.width() - 10));
            painter->setFont(keyFont);
        }
    };
    drawRows(topParameters_, 18.0);
    drawRows(bottomParameters_, height_ - bottomParameters_.size() * 25.0 - 18.0);
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
            const qreal y = output ? height_ - 9.0 : 9.0;
            candidate->setPos(x, y);
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
