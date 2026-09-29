#include "GraphView.hpp"

#include "GraphScene.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QScrollBar>
#include <QWheelEvent>

GraphView::GraphView(GraphScene *scene, QWidget *parent) : QGraphicsView(scene, parent) {
    setRenderHint(QPainter::Antialiasing);
    setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    setTransformationAnchor(QGraphicsView::NoAnchor);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setDragMode(QGraphicsView::RubberBandDrag);
    setRubberBandSelectionMode(Qt::IntersectsItemShape);
    setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
    setBackgroundBrush(QColor(250, 251, 253));
    setFrameShape(QFrame::NoFrame);
    setFocusPolicy(Qt::StrongFocus);
}

void GraphView::fitGraph() {
    QRectF bounds = sceneRect();
    if (scene() && !scene()->items().isEmpty()) bounds = scene()->itemsBoundingRect().adjusted(-50, -50, 50, 50);
    if (bounds.isEmpty()) return;
    fitInView(bounds, Qt::KeepAspectRatio);
    zoom_ = qBound<qreal>(0.2, transform().m11(), 3.0);
    setTransform(QTransform::fromScale(zoom_, zoom_));
}

void GraphView::wheelEvent(QWheelEvent *event) {
    const qreal factor = event->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
    const qreal next = qBound<qreal>(0.2, zoom_ * factor, 3.0);
    if (qFuzzyCompare(next, zoom_)) return;
    zoom_ = next;
    const QGraphicsView::ViewportAnchor previousAnchor = transformationAnchor();
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    scale(factor, factor);
    setTransformationAnchor(previousAnchor);
    event->accept();
}

void GraphView::mousePressEvent(QMouseEvent *event) {
    if (event->button() == Qt::MiddleButton || (spaceDown_ && event->button() == Qt::LeftButton)) {
        panning_ = true;
        lastPan_ = event->pos();
        viewport()->setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }
    QGraphicsView::mousePressEvent(event);
}

void GraphView::mouseMoveEvent(QMouseEvent *event) {
    if (panning_) {
        const QPoint delta = event->pos() - lastPan_;
        lastPan_ = event->pos();
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
        event->accept();
        return;
    }
    QGraphicsView::mouseMoveEvent(event);
}

void GraphView::mouseReleaseEvent(QMouseEvent *event) {
    if (panning_ && (event->button() == Qt::MiddleButton || event->button() == Qt::LeftButton)) {
        panning_ = false;
        viewport()->unsetCursor();
        event->accept();
        return;
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void GraphView::keyPressEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Escape) {
        if (auto *graph = qobject_cast<GraphScene *>(scene())) graph->cancelInteraction();
        event->accept();
        return;
    }
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spaceDown_ = true;
        event->accept();
        return;
    }
    QGraphicsView::keyPressEvent(event);
}

void GraphView::keyReleaseEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Space && !event->isAutoRepeat()) {
        spaceDown_ = false;
        event->accept();
        return;
    }
    QGraphicsView::keyReleaseEvent(event);
}
