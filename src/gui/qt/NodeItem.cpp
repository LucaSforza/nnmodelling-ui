#include "NodeItem.hpp"

#include "GraphScene.hpp"
#include "PortItem.hpp"
#include "model/model.h"

#include <QBrush>
#include <QFont>
#include <QFontMetricsF>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsSceneHoverEvent>
#include <QGraphicsRectItem>
#include <QGraphicsSimpleTextItem>
#include <algorithm>
#include <cmath>
#include <QPainter>
#include <QPainterPath>
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
    contentWidth_ = required;
    setPos(position);
    setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsGeometryChanges);
    setAcceptHoverEvents(true);
    setCacheMode(DeviceCoordinateCache);
    setZValue(1.0);
}

QRectF NodeItem::boundingRect() const { return QRectF(0.5, 0.5, width_ - 1, height_ - 1); }

void NodeItem::setBoundaryKind(const QString &kind) {
    prepareGeometryChange();
    boundaryKind_ = kind;
    if (!kind.isEmpty()) {
        QFont labelFont;
        labelFont.setBold(true);
        labelFont.setPointSizeF(11.0);
        width_ = qMax<qreal>(190.0, 100.0 + QFontMetricsF(labelFont).horizontalAdvance(label_));
        height_ = 76.0;
    }
    update();
}

void NodeItem::setFlowDirection(FlowDirection direction) {
    if (flowDirection_ != direction) prepareGeometryChange();
    flowDirection_ = direction;
    if (joinNode_) {
        layoutJoin();
        update();
        return;
    }
    layoutPorts();
    update();
}

void NodeItem::setJoinNode(bool join) {
    if (joinNode_ == join) return;
    prepareGeometryChange();
    joinNode_ = join;
    if (joinNode_) layoutJoin();
    update();
}

void NodeItem::setJoinInputControls(bool canRemove, bool canAdd) {
    canRemoveJoinInput_ = canRemove;
    canAddJoinInput_ = canAdd;
    update();
}

void NodeItem::layoutJoin() {
    const int inputCount = std::count_if(ports_.cbegin(), ports_.cend(), [](PortItem *port) {
        return !port->isOutput();
    });
    const int outputCount = ports_.size() - inputCount;
    const int spreadCount = qMax(inputCount, outputCount);
    const int parameterCount = topParameters_.size() + bottomParameters_.size();
    if (flowDirection_ == FlowDirection::Vertical) {
        width_ = qMax<qreal>(contentWidth_, qMax<qreal>(220.0, spreadCount * 20.0 + 48.0));
        height_ = qMax<qreal>(170.0, 145.0 + parameterCount * 24.0);
        junctionRect_ = QRectF(width_ / 2.0 - 50.0, 58.0, 100.0, 18.0);
        joinRemoveRect_ = QRectF(junctionRect_.left() - 30.0, junctionRect_.center().y() - 11.0, 22.0, 22.0);
        joinAddRect_ = QRectF(junctionRect_.right() + 8.0, junctionRect_.center().y() - 11.0, 22.0, 22.0);
    } else {
        width_ = qMax<qreal>(contentWidth_, qMax<qreal>(260.0, 236.0 + 20.0 * parameterCount));
        const qreal barHeight = qMax<qreal>(60.0, spreadCount * 20.0 + 20.0);
        height_ = barHeight + 100.0 + parameterCount * 24.0;
        junctionRect_ = QRectF(width_ / 2.0 - 9.0, 34.0, 18.0, barHeight);
        joinRemoveRect_ = QRectF(junctionRect_.center().x() - 11.0,
                                 junctionRect_.top() - 28.0, 22.0, 22.0);
        joinAddRect_ = QRectF(junctionRect_.center().x() - 11.0,
                              junctionRect_.bottom() + 6.0, 22.0, 22.0);
    }
    layoutPorts();
}

void NodeItem::layoutPorts() {
    for (bool output : {true, false}) {
        int order = 0;
        const int count = std::count_if(ports_.cbegin(), ports_.cend(), [output](PortItem *p) {
            return p->isOutput() == output;
        });
        for (PortItem *port : ports_) {
            if (port->isOutput() != output) continue;
            if (!boundaryKind_.isEmpty()) {
                const qreal centerX = 47.0, centerY = 38.0, radius = 22.0;
                const bool horizontal = flowDirection_ == FlowDirection::Horizontal;
                const qreal spread = count <= 1 ? 0.0 : (order - (count - 1) / 2.0) * 0.75;
                const qreal angle = horizontal
                    ? (output ? 3.14159265358979323846 + spread : spread)
                    : (output ? 1.57079632679489661923 + spread
                              : -1.57079632679489661923 + spread);
                port->setPos(centerX + radius * std::cos(angle),
                             centerY + radius * std::sin(angle));
                port->setBoundaryHandlePresentation(true);
            } else if (joinNode_) {
                if (flowDirection_ == FlowDirection::Vertical) {
                    const qreal centerX = width_ / 2.0;
                    const qreal span = width_ - 48.0;
                    port->setPos(centerX + (count <= 1 ? 0.0 :
                        (order - (count - 1) / 2.0) * (span / (count - 1))),
                        output ? junctionRect_.bottom() + 24.0 : junctionRect_.top() - 22.0);
                } else {
                    const qreal span = junctionRect_.height() - 24.0;
                    port->setPos(junctionRect_.center().x() + (output ? -24.0 : 24.0),
                                 junctionRect_.center().y() + (count <= 1 ? 0.0 :
                                    (order - (count - 1) / 2.0) * span / (count - 1)));
                }
            } else if (flowDirection_ == FlowDirection::Vertical) {
                port->setPos(width_ * (order + 1.0) / (count + 1.0), output ? height_ - 9.0 : 9.0);
            } else {
                port->setPos(output ? 9.0 : width_ - 9.0,
                             height_ * (order + 1.0) / (count + 1.0));
            }
            ++order;
        }
    }
    update();
}

void NodeItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) {
    painter->setRenderHint(QPainter::Antialiasing);
    if (joinNode_) {
        if (isSelected()) {
            painter->setPen(QPen(QColor("#3978c5"), 2.0));
            painter->setBrush(Qt::NoBrush);
            painter->drawRoundedRect(boundingRect().adjusted(2.0, 2.0, -2.0, -2.0), 8.0, 8.0);
        }
        painter->setPen(Qt::NoPen);
        painter->setBrush(QColor(32, 39, 48));
        painter->drawRoundedRect(junctionRect_, 2.0, 2.0);

        const auto drawControl = [painter](const QRectF &rect, const QString &symbol, bool enabled) {
            painter->setPen(QPen(enabled ? QColor(95, 108, 124) : QColor(172, 178, 185), 1.0));
            painter->setBrush(enabled ? QColor(246, 248, 250) : QColor(229, 232, 235));
            painter->drawEllipse(rect);
            painter->setPen(enabled ? QColor(43, 54, 67) : QColor(149, 155, 162));
            QFont font = painter->font();
            font.setBold(true);
            font.setPointSizeF(10.0);
            painter->setFont(font);
            painter->drawText(rect, Qt::AlignCenter, symbol);
        };
        drawControl(joinRemoveRect_, QStringLiteral("−"), canRemoveJoinInput_);
        drawControl(joinAddRect_, QStringLiteral("+"), canAddJoinInput_);

        QFont portFont = painter->font();
        portFont.setPointSizeF(8.0);
        painter->setFont(portFont);
        painter->setPen(QColor(74, 86, 99));
        for (PortItem *port : ports_) {
            if (flowDirection_ == FlowDirection::Vertical) {
                const qreal y = port->isOutput() ? port->y() + 5.0 : port->y() - 16.0;
                painter->drawText(QRectF(port->x() - 45.0, y, 90.0, 12.0),
                    Qt::AlignHCenter | Qt::AlignVCenter, port->label());
            } else {
                const QRectF label = port->isOutput()
                    ? QRectF(5.0, port->y() - 7.0, port->x() - 17.0, 14.0)
                    : QRectF(port->x() + 7.0, port->y() - 7.0, width_ - port->x() - 12.0, 14.0);
                painter->drawText(label, port->isOutput() ? Qt::AlignRight | Qt::AlignVCenter
                                                         : Qt::AlignLeft | Qt::AlignVCenter,
                                  port->label());
            }
        }

        QFont titleFont = painter->font();
        titleFont.setBold(true);
        titleFont.setPointSizeF(11.0);
        painter->setFont(titleFont);
        painter->setPen(QColor(38, 52, 70));
        const qreal titleY = flowDirection_ == FlowDirection::Vertical
            ? junctionRect_.bottom() + 48.0 : junctionRect_.bottom() + 39.0;
        painter->drawText(QRectF(8.0, titleY, width_ - 16.0, 20.0),
                          Qt::AlignHCenter | Qt::AlignVCenter, label_);
        const int parameterCount = topParameters_.size() + bottomParameters_.size();
        const qreal parameterY = height_ - parameterCount * 22.0 - 8.0;
        QFont parameterFont = painter->font();
        parameterFont.setBold(false);
        parameterFont.setPointSizeF(9.0);
        painter->setFont(parameterFont);
        int row = 0;
        for (const auto &entry : topParameters_ + bottomParameters_) {
            const qreal y = parameterY + row++ * 22.0;
            painter->setPen(QColor(74, 86, 99));
            painter->drawText(QRectF(12.0, y, width_ * 0.55, 18.0),
                              Qt::AlignLeft | Qt::AlignVCenter, entry.first);
            painter->setPen(QColor(38, 52, 70));
            painter->drawText(QRectF(width_ * 0.57, y, width_ * 0.4, 18.0),
                              Qt::AlignRight | Qt::AlignVCenter, entry.second);
        }
        if (!problemCategory_.isEmpty()) {
            QColor marker = problemCategory_ == QStringLiteral("lua-compilation") ? QColor("#9c3d79")
                : problemCategory_ == QStringLiteral("incomplete") ? QColor("#a66a12")
                : problemCategory_ == QStringLiteral("internal") ? QColor("#554d79")
                : QColor("#b23b35");
            painter->setPen(Qt::NoPen);
            painter->setBrush(marker);
            painter->drawEllipse(QRectF(2.0, 2.0, 16.0, 16.0));
            painter->setPen(Qt::white);
            QFont badge = painter->font();
            badge.setBold(true);
            badge.setPointSizeF(8.0);
            painter->setFont(badge);
            const QString symbol = problemCategory_ == QStringLiteral("incomplete")
                ? QStringLiteral("?")
                : problemCategory_ == QStringLiteral("lua-compilation") ? QStringLiteral("L")
                : problemCategory_ == QStringLiteral("internal") ? QStringLiteral("×")
                : QStringLiteral("!");
            painter->drawText(QRectF(2.0, 2.0, 16.0, 16.0), Qt::AlignCenter, symbol);
        }
        return;
    }
    if (!boundaryKind_.isEmpty()) {
        const QColor fill = boundaryKind_ == QStringLiteral("input") ? QColor("#111111")
            : boundaryKind_ == QStringLiteral("output") ? QColor("#8b5a2b") : QColor("#c62828");
        const QRectF circle(25.0, 16.0, 44.0, 44.0);
        if (isSelected()) {
            painter->setPen(QPen(QColor("#3978c5"), 4.0));
            painter->setBrush(Qt::NoBrush);
            painter->drawEllipse(circle.adjusted(-5, -5, 5, 5));
        }
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
            const QRectF badge(5.0, 3.0, 17.0, 17.0);
            painter->setPen(Qt::NoPen);
            painter->setBrush(marker);
            painter->drawRoundedRect(badge, 4.0, 4.0);
            painter->setPen(Qt::white);
            QFont badgeFont = painter->font();
            badgeFont.setBold(true);
            badgeFont.setPointSizeF(8.0);
            painter->setFont(badgeFont);
            painter->drawText(badge, Qt::AlignCenter, symbol);
        }
        painter->setPen(Qt::NoPen);
        painter->setBrush(fill);
        painter->drawEllipse(circle);
        painter->setPen(QColor("#263446"));
        QFont labelFont = painter->font();
        labelFont.setBold(true);
        labelFont.setPointSizeF(11.0);
        painter->setFont(labelFont);
        const QFontMetricsF labelMetrics(labelFont);
        const bool inputBoundary = boundaryKind_ == QStringLiteral("input");
        const QRectF labelRect(82.0, inputBoundary ? 2.0 : 15.0,
                               width_ - 94.0,
                               inputBoundary ? labelMetrics.height() + 2.0 : height_ - 20.0);
        painter->drawText(labelRect,
                          Qt::AlignLeft | Qt::AlignVCenter, label_);
        return;
    }
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
    QPainterPath cardClip;
    cardClip.addRoundedRect(card, 12, 12);
    painter->save();
    painter->setClipPath(cardClip);
    for (int i = 0; i < topParameters_.size(); ++i) {
        const qreal y = 9.5 + i * bandHeight;
        painter->fillRect(QRectF(card.left(), y, card.width(), bandHeight), band);
        painter->setPen(QPen(QColor(25, 42, 59, 105), 1.0));
        painter->drawLine(QPointF(card.left(), y + bandHeight),
                          QPointF(card.right(), y + bandHeight));
    }
    for (int i = 0; i < bottomParameters_.size(); ++i) {
        const qreal y = height_ - 9.5 - (i + 1) * bandHeight;
        painter->fillRect(QRectF(card.left(), y, card.width(), bandHeight), band);
        painter->setPen(QPen(QColor(25, 42, 59, 105), 1.0));
        painter->drawLine(QPointF(card.left(), y), QPointF(card.right(), y));
    }
    painter->restore();
    painter->setBrush(Qt::NoBrush);
    painter->setPen(QPen(isSelected() ? QColor(30, 83, 142) : QColor(40, 49, 59),
                         isSelected() ? 3.2 : 3.0));
    painter->drawRoundedRect(card, 12, 12);

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
    if (externalPortsOnBoundary_) return;
    if (!boundaryKind_.isEmpty()) return;
    for (PortItem *port : ports_) {
        const bool vertical = flowDirection_ == FlowDirection::Vertical;
        const QRectF labelRect = vertical
            ? QRectF(port->x() - 46.0, port->isOutput() ? height_ - 21.0 : 0.0, 92.0, 10.0)
            : port->isOutput()
                ? QRectF(14.0, port->y() - 5.0, width_ / 2.0 - 18.0, 10.0)
                : QRectF(width_ / 2.0, port->y() - 5.0, width_ / 2.0 - 14.0, 10.0);
        painter->drawText(labelRect, vertical ? Qt::AlignHCenter | Qt::AlignVCenter
                                              : (port->isOutput() ? Qt::AlignRight : Qt::AlignLeft) |
                                                    Qt::AlignVCenter,
                          port->label());
    }
}

void NodeItem::addPort(PortItem *port) {
    ports_.append(port);
    if (joinNode_) {
        prepareGeometryChange();
        layoutJoin();
    } else {
        layoutPorts();
    }
    update();
}

void NodeItem::setChildCount(int) {}

void NodeItem::setReadOnlyPreview(bool preview) {
    readOnlyPreview_ = preview;
    setFlag(ItemIsMovable, !preview);
    setFlag(ItemIsSelectable, !preview);
    for (PortItem *port : ports_) port->setReadOnlyPreview(preview);
}

void NodeItem::setTensorSummary(const QString &summary) {
    tensorSummary_ = summary;
    if (!tensorSummary_.isEmpty() && !tensorPopupBackground_) {
        tensorPopupBackground_ = new QGraphicsRectItem(this);
        tensorPopupBackground_->setBrush(QColor(255, 255, 255, 245));
        tensorPopupBackground_->setPen(QPen(QColor(142, 155, 169), 1.0));
        tensorPopupBackground_->setZValue(1000.0);
        tensorPopupBackground_->setAcceptedMouseButtons(Qt::NoButton);
        tensorPopupBackground_->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        tensorPopupText_ = new QGraphicsSimpleTextItem(tensorPopupBackground_);
        tensorPopupText_->setBrush(QColor(36, 48, 62));
        QFont font = tensorPopupText_->font();
        font.setPointSizeF(9.0);
        tensorPopupText_->setFont(font);
        tensorPopupBackground_->hide();
    }
    if (tensorPopupBackground_) {
        if (tensorSummary_.isEmpty()) {
            tensorPopupBackground_->hide();
            tensorPopupText_->setText(QString());
        } else {
            tensorPopupText_->setText(tensorSummary_);
            const QRectF textRect = tensorPopupText_->boundingRect();
            tensorPopupText_->setPos(8, 6);
            tensorPopupBackground_->setRect(0, 0, textRect.width() + 16, textRect.height() + 12);
            tensorPopupBackground_->setPos(boundingRect().right() + 12, 4);
        }
    }
    setToolTip(QString());
}

void NodeItem::setRoutingWarning(bool warning) {
    if (warning && !routingWarningItem_) {
        routingWarningItem_ = new QGraphicsSimpleTextItem(QStringLiteral("!"), this);
        QFont font = routingWarningItem_->font();
        font.setBold(true);
        font.setPointSizeF(13.0);
        routingWarningItem_->setFont(font);
        routingWarningItem_->setBrush(QColor("#bd5a20"));
        routingWarningItem_->setToolTip(tr("Routing blocked. Move overlapping nodes or Arrange."));
        routingWarningItem_->setAcceptedMouseButtons(Qt::NoButton);
        routingWarningItem_->setFlag(QGraphicsItem::ItemIgnoresTransformations);
        routingWarningItem_->setZValue(10.0);
    }
    if (routingWarningItem_) {
        routingWarningItem_->setVisible(warning);
        routingWarningItem_->setPos(boundingRect().right() - 17.0, 0.0);
    }
}

void NodeItem::setTensorPopupVisible(bool visible) {
    if (tensorPopupBackground_ && !tensorSummary_.isEmpty())
        tensorPopupBackground_->setVisible(visible);
}

void NodeItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
    if (tensorHoverRegion().contains(event->pos())) setTensorPopupVisible(true);
    setZValue(1000.0);
    for (QGraphicsItem *parent = parentItem(); parent; parent = parent->parentItem()) {
        parent->setZValue(1000.0);
        if (auto *node = dynamic_cast<NodeItem *>(parent)) node->setTensorPopupVisible(false);
    }
    QGraphicsObject::hoverEnterEvent(event);
}

void NodeItem::hoverMoveEvent(QGraphicsSceneHoverEvent *event) {
    setTensorPopupVisible(tensorHoverRegion().contains(event->pos()));
    QGraphicsObject::hoverMoveEvent(event);
}

void NodeItem::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
    setTensorPopupVisible(false);
    setZValue(1.0);
    for (QGraphicsItem *parent = parentItem(); parent; parent = parent->parentItem())
        parent->setZValue(1.0);
    QGraphicsObject::hoverLeaveEvent(event);
}

void NodeItem::setProblemCategory(const QString &category) {
    if (problemCategory_ == category) return;
    problemCategory_ = category;
    update();
}

QVariant NodeItem::itemChange(GraphicsItemChange change, const QVariant &value) {
    if (change == ItemPositionChange && owner_ && !owner_->refreshing_) {
        const QPointF position = value.toPointF();
        constexpr qreal grid = NN_MODEL_GRID_SPACING;
        return QPointF(qRound(position.x() / grid) * grid,
                       qRound(position.y() / grid) * grid);
    }
    if (change == ItemPositionHasChanged && owner_)
        owner_->updateEdgesForNode(id_);
    return QGraphicsObject::itemChange(change, value);
}

void NodeItem::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    if (readOnlyPreview_) { event->accept(); return; }
    if (joinNode_ && owner_ && event->button() == Qt::LeftButton) {
        if (joinRemoveRect_.contains(event->pos())) {
            owner_->adjustJoinInputSlots(id_, false);
            event->accept();
            return;
        }
        if (joinAddRect_.contains(event->pos())) {
            owner_->adjustJoinInputSlots(id_, true);
            event->accept();
            return;
        }
    }
    QGraphicsObject::mousePressEvent(event);
    if (owner_) owner_->beginNodeDrag(this);
}

void NodeItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
    if (readOnlyPreview_) { event->accept(); return; }
    QGraphicsObject::mouseReleaseEvent(event);
    if (owner_) owner_->commitNodeMoves();
}
