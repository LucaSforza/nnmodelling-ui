#include "NodeItem.hpp"

#include "GraphScene.hpp"
#include "PortItem.hpp"

#include <QBrush>
#include <QFont>
#include <QFontMetricsF>
#include <QGraphicsSceneMouseEvent>
#include <algorithm>
#include <cmath>
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
    const QRectF card(1.5, 9.5, width_ - 3.0, height_ - 19.0);
    const QColor face = color_.darker(125);
    const QColor band = color_.darker(145);
    const auto foregroundFor = [](const QColor &color) {
        const auto channel = [](int value) {
            const qreal c = value / 255.0;
            return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
        };
        const qreal luminance = 0.2126 * channel(color.red()) +
                                0.7152 * channel(color.green()) +
                                0.0722 * channel(color.blue());
        return 1.05 / (luminance + 0.05) >=
                       (luminance + 0.05) / (0.015 + 0.05)
            ? QColor(Qt::white) : QColor(27, 39, 53);
    };
    painter->setPen(QPen(isSelected() ? QColor(30, 83, 142) : QColor(40, 49, 59),
                         isSelected() ? 3.2 : 3.0));
    painter->setBrush(face);
    painter->drawRoundedRect(card, 12, 12);

    const qreal bandHeight = 25.0;
    for (int i = 0; i < topParameters_.size(); ++i) {
        const qreal y = 9.5 + i * bandHeight;
        painter->fillRect(QRectF(4.5, y, width_ - 9.0, bandHeight), band);
        painter->setPen(QPen(QColor(25, 42, 59, 105), 1.0));
        painter->drawLine(QPointF(4.5, y + bandHeight), QPointF(width_ - 4.5, y + bandHeight));
    }
    for (int i = 0; i < bottomParameters_.size(); ++i) {
        const qreal y = height_ - 9.5 - (i + 1) * bandHeight;
        painter->fillRect(QRectF(4.5, y, width_ - 9.0, bandHeight), band);
        painter->setPen(QPen(QColor(25, 42, 59, 105), 1.0));
        painter->drawLine(QPointF(4.5, y), QPointF(width_ - 4.5, y));
    }

    painter->setPen(foregroundFor(face));
    QFont title = painter->font();
    title.setBold(true);
    title.setPointSizeF(16.0);
    painter->setFont(title);
    QFontMetricsF metrics(title);
    const qreal centerTop = 9.5 + topParameters_.size() * bandHeight;
    const qreal centerBottom = height_ - 9.5 - bottomParameters_.size() * bandHeight;
    const qreal titleY = centerTop + (centerBottom - centerTop - 32.0) / 2.0;
    qreal titleX = 12.0;
    qreal titleWidth = width_ - 24.0;
    if (!problemCategory_.isEmpty()) {
        QColor marker = QColor("#b23b35");
        QString symbol = QStringLiteral("!");
        if (problemCategory_ == QStringLiteral("lua-compilation")) {
            marker = QColor("#9c3d79");
            symbol = QStringLiteral("L");
        } else if (problemCategory_ == QStringLiteral("incomplete")) {
            marker = QColor("#a66a12");
            symbol = QStringLiteral("?");
        } else if (problemCategory_ == QStringLiteral("internal")) {
            marker = QColor("#554d79");
            symbol = QStringLiteral("×");
        }
        const QRectF markerRect(12, titleY + 2, 14, 14);
        painter->setPen(Qt::NoPen);
        painter->setBrush(marker);
        painter->drawEllipse(markerRect);
        painter->setPen(Qt::white);
        QFont markerFont = title;
        markerFont.setPointSizeF(8.0);
        painter->setFont(markerFont);
        painter->drawText(markerRect, Qt::AlignCenter, symbol);
        painter->setPen(QColor(37, 49, 64));
        titleX = 31.0;
        titleWidth -= 17.0;
    }
    const QString displayLabel = metrics.elidedText(label_, Qt::ElideRight, titleWidth);
    painter->setFont(title);
    painter->drawText(QRectF(titleX, titleY, titleWidth, 32), Qt::AlignHCenter | Qt::AlignVCenter,
                      displayLabel);
    auto drawRows = [painter, this, bandHeight, foregroundFor, face, band](
                        const QList<QPair<QString, QString>> &rows, qreal firstY) {
        QFont keyFont = painter->font();
        keyFont.setBold(false);
        keyFont.setPointSizeF(10.5);
        QFont valueFont = keyFont;
        valueFont.setBold(true);
        painter->setFont(keyFont);
        for (int i = 0; i < rows.size(); ++i) {
            const qreal y = firstY + i * bandHeight;
            const QRectF row(11, y, width_ - 22, bandHeight);
            const auto &entry = rows[i];
            const qreal badgeWidth = qMin(row.width() - 48.0,
                QFontMetricsF(valueFont).horizontalAdvance(entry.second) + 18.0);
            const QRectF badge(row.right() - badgeWidth, row.top() + 2.0,
                               badgeWidth, row.height() - 4.0);
            painter->setPen(foregroundFor(band));
            painter->drawText(QRectF(row.left() + 5, row.top(),
                                     row.width() - badgeWidth - 12, row.height()),
                              Qt::AlignLeft | Qt::AlignVCenter,
                              QFontMetricsF(keyFont).elidedText(entry.first, Qt::ElideRight,
                                  row.width() - badgeWidth - 17));
            painter->setPen(Qt::NoPen);
            painter->setBrush(face.darker(112));
            painter->drawRoundedRect(badge, 7, 7);
            painter->setPen(foregroundFor(face.darker(112)));
            painter->setFont(valueFont);
            painter->drawText(badge.adjusted(7, 0, -7, 0), Qt::AlignRight | Qt::AlignVCenter,
                QFontMetricsF(valueFont).elidedText(entry.second, Qt::ElideLeft, badge.width() - 10));
            painter->setFont(keyFont);
        }
    };
    drawRows(topParameters_, 9.5);
    drawRows(bottomParameters_, height_ - bottomParameters_.size() * bandHeight - 9.5);
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

void NodeItem::setProblemCategory(const QString &category) {
    if (problemCategory_ == category) return;
    problemCategory_ = category;
    update();
}

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
