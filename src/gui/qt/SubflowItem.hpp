#ifndef NN_SUBFLOW_ITEM_HPP
#define NN_SUBFLOW_ITEM_HPP

#include "NodeItem.hpp"
#include <QRectF>

class SubflowItem final : public NodeItem {
    Q_OBJECT
public:
    SubflowItem(GraphScene *owner, QString id, QString label, QString packageId,
                const QPointF &position, const QColor &color,
                QList<QPair<QString, QString>> topParameters = {},
                QList<QPair<QString, QString>> bottomParameters = {});
    void setChildCount(int count) override;
    QRectF boundingRect() const override;
    void setFlowDirection(FlowDirection direction) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
private:
    int childCount_ = 0;
    QRectF footerRect() const;
};

#endif
