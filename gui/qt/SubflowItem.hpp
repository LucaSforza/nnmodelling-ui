#ifndef NN_SUBFLOW_ITEM_HPP
#define NN_SUBFLOW_ITEM_HPP

#include "NodeItem.hpp"

class SubflowItem final : public NodeItem {
    Q_OBJECT
public:
    using NodeItem::NodeItem;
    void setChildCount(int count) override;
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
               QWidget *widget = nullptr) override;

protected:
    void mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) override;
private:
    int childCount_ = 0;
};

#endif
