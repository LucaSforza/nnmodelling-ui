#include "Network3DView.hpp"

#include <QFocusEvent>
#include <QApplication>
#include <QResizeEvent>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QToolButton>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iterator>

Network3DView::Network3DView(QWidget *parent) : QWidget(parent) {
    setObjectName(QStringLiteral("network3DView"));
    setFocusPolicy(Qt::StrongFocus);
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    auto *toolbar = new QHBoxLayout;
    auto *instructions = new QLabel(
        tr("Arrows: turn  ·  Right-drag: look  ·  WASD: fly  ·  Q/E: down/up  ·  Shift: faster  ·  Wheel: fly"), this);
    instructions->setObjectName(QStringLiteral("network3DInstructions"));
    instructions->setTextInteractionFlags(Qt::TextSelectableByMouse);
    toolbar->addWidget(instructions, 1);
    fitButton_ = new QToolButton(this);
    fitButton_->setObjectName(QStringLiteral("network3DFit"));
    fitButton_->setText(tr("Fit whole graph"));
    toolbar->addWidget(fitButton_);
    homeButton_ = new QToolButton(this);
    homeButton_->setObjectName(QStringLiteral("network3DHome"));
    homeButton_->setText(tr("Home / start"));
    toolbar->addWidget(homeButton_);
    layout->addLayout(toolbar);

    error_ = new QLabel(this);
    error_->setObjectName(QStringLiteral("network3DError"));
    error_->setStyleSheet(QStringLiteral("color: #9e2a2b; padding: 2px 6px;"));
    error_->setWordWrap(true);
    error_->hide();
    layout->addWidget(error_);
    layout->addStretch(1);
    selection_ = new QLabel(this);
    selection_->setObjectName(QStringLiteral("network3DSelection"));
    selection_->setWordWrap(true);
    selection_->setTextInteractionFlags(Qt::TextSelectableByMouse);
    selection_->setMinimumHeight(46);
    selection_->setMaximumHeight(110);
    selection_->setStyleSheet(QStringLiteral("background: #f4f6f8; padding: 6px; border: 1px solid #c5cdd6;"));
    selection_->setText(tr("Select an occurrence to inspect its source and parameters."));
    layout->addWidget(selection_);
    movementTimer_ = new QTimer(this);
    movementTimer_->setInterval(16);
    connect(movementTimer_, &QTimer::timeout, this, &Network3DView::tickMovement);
    connect(fitButton_, &QToolButton::clicked, this, &Network3DView::fitWholeGraph);
    connect(homeButton_, &QToolButton::clicked, this, &Network3DView::homeCamera);
    setStyleSheet(QStringLiteral("Network3DView { background: white; }"));
}

Network3DView::~Network3DView() {
    nn_3d_frame_dispose(&frame_);
    nn_3d_free(scene_);
}

bool Network3DView::rebuild(const NNProject *project) {
    nn_3d_frame_dispose(&frame_);
    nn_3d_free(scene_);
    scene_ = nullptr;
    selectedPath_.clear();
    selection_->setText(tr("Select an occurrence to inspect its source and parameters."));
    selection_->setToolTip(QString());
    error_->clear();
    error_->hide();
    if (project) {
        char message[512] = {};
        scene_ = nn_3d_build(project, message, sizeof(message));
        if (!scene_) {
            error_->setText(QString::fromUtf8(message[0] ? message : "Could not build the 3D scene."));
            error_->show();
        }
    } else {
        error_->setText(tr("Open or create a project to explore its network."));
        error_->show();
    }
    layout()->activate();
    refreshCameraAspect();
    updateFrame();
    emit sceneReady(scene_ != nullptr);
    return scene_ != nullptr;
}

QString Network3DView::selectedPath() const { return selectedPath_; }
QRect Network3DView::sceneViewportRect() const {
    const auto *layout = qobject_cast<const QVBoxLayout *>(this->layout());
    if (!layout || layout->count() < 4) return {};
    const QLayout *toolbar = layout->itemAt(0)->layout();
    const QWidget *error = layout->itemAt(1)->widget();
    const QWidget *selection = layout->itemAt(3)->widget();
    if (!toolbar || !selection) return {};
    int top = toolbar->geometry().bottom() + 1;
    if (error && !error->isHidden()) top = error->geometry().bottom() + 1;
    return QRect(0, top, width(), qMax(0, selection->geometry().top() - top));
}

void Network3DView::fitWholeGraph() {
    if (!scene_) return;
    const QRect canvas = sceneViewportRect();
    nn_3d_camera_fit(scene_, &camera_, qMax(0.1, canvas.width() / double(qMax(1, canvas.height()))));
    updateFrame();
}

void Network3DView::homeCamera() {
    if (!scene_) return;
    const QRect canvas = sceneViewportRect();
    nn_3d_camera_home(scene_, &camera_, qMax(0.1, canvas.width() / double(qMax(1, canvas.height()))));
    updateFrame();
}

void Network3DView::updateFrame() {
    nn_3d_frame_dispose(&frame_);
    if (error_->text() == tr("Could not project the 3D scene.")) {
        error_->clear();
        error_->hide();
        layout()->activate();
    }
    if (scene_) {
        const QRect canvas = sceneViewportRect();
        if (canvas.width() > 0 && canvas.height() > 0 &&
            !nn_3d_frame(scene_, &camera_, canvas.width(), canvas.height(), &frame_)) {
            nn_3d_frame_dispose(&frame_);
            error_->setText(tr("Could not project the 3D scene."));
            error_->show();
            layout()->activate();
        }
    }
    update();
}

void Network3DView::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), Qt::white);
    const QRect canvas = sceneViewportRect();
    painter.save();
    painter.setClipRect(canvas);
    painter.translate(canvas.topLeft());
    for (size_t i = 0; i < frame_.count; ++i) {
        const NN3DPrimitive &primitive = frame_.items[i];
        const QColor color = QColor::fromRgba(primitive.rgba);
        painter.setPen(QPen(color, primitive.kind == NN_3D_LINE ? 1.4 : 1.0));
        if (primitive.kind == NN_3D_FACE) {
            QPolygonF polygon;
            for (int p = 0; p < 4; ++p) polygon << QPointF(primitive.x[p], primitive.y[p]);
            painter.setBrush(color);
            painter.drawPolygon(polygon);
        } else if (primitive.kind == NN_3D_LINE) {
            painter.setBrush(Qt::NoBrush);
            painter.drawLine(QPointF(primitive.x[0], primitive.y[0]), QPointF(primitive.x[1], primitive.y[1]));
        } else if (primitive.text) {
            painter.setBrush(Qt::NoBrush);
            painter.setPen(QColor(QStringLiteral("#172536")));
            painter.drawText(QPointF(primitive.x[0], primitive.y[0]), QString::fromUtf8(primitive.text));
        }
    }
    if (!scene_) {
        painter.setPen(QColor(QStringLiteral("#637184")));
        painter.drawText(QRect(20, 30, canvas.width() - 40, 80), Qt::AlignCenter, tr("Network 3D scene unavailable"));
    }
    painter.restore();
}

void Network3DView::pickAt(const QPointF &position) {
    if (!scene_ || !frame_.items) return;
    const QRect canvas = sceneViewportRect();
    if (!canvas.contains(position.toPoint())) return;
    const QPointF local = position - canvas.topLeft();
    const size_t picked = nn_3d_pick(&frame_, local.x(), local.y());
    if (picked != std::numeric_limits<size_t>::max()) updateSelection(picked);
}

void Network3DView::updateSelection(size_t index) {
    const NN3DNode *node = scene_ ? nn_3d_node_at(scene_, index) : nullptr;
    if (!node) return;
    selectedPath_ = QString::fromUtf8(node->path ? node->path : "");
    const QString package = QString::fromUtf8(node->package_id ? node->package_id : "");
    const QString label = QString::fromUtf8(node->label ? node->label : "");
    QString details = tr("%1  ·  source %2  ·  %3").arg(label, QString::fromUtf8(node->source_id ? node->source_id : ""), package);
    details += tr("\nOccurrence: %1").arg(selectedPath_);
    if (node->parameters && *node->parameters)
        details += tr("\nParameters: %1").arg(QString::fromUtf8(node->parameters));
    selection_->setText(details);
    selection_->setToolTip(details);
    layout()->activate();
    updateFrame();
}

void Network3DView::mousePressEvent(QMouseEvent *event) {
    setFocus(Qt::MouseFocusReason);
    if (event->button() == Qt::RightButton) {
        rightDragging_ = true;
        lastMouse_ = event->position().toPoint();
        event->accept();
        return;
    }
    if (event->button() == Qt::LeftButton) pickAt(event->position());
    QWidget::mousePressEvent(event);
}

void Network3DView::mouseMoveEvent(QMouseEvent *event) {
    if (rightDragging_ && scene_) {
        const QPoint current = event->position().toPoint();
        const QPoint delta = current - lastMouse_;
        lastMouse_ = current;
        nn_3d_camera_look(&camera_, -delta.x() * 0.006, -delta.y() * 0.006);
        updateFrame();
        event->accept();
        return;
    }
    QWidget::mouseMoveEvent(event);
}

void Network3DView::mouseReleaseEvent(QMouseEvent *event) {
    if (event->button() == Qt::RightButton) {
        rightDragging_ = false;
        event->accept();
        return;
    }
    QWidget::mouseReleaseEvent(event);
}

void Network3DView::wheelEvent(QWheelEvent *event) {
    if (scene_) {
        const double wheelDistance = event->modifiers() & Qt::ShiftModifier ? 160.0 : 32.0;
        nn_3d_camera_move(&camera_, 0, 0, event->angleDelta().y() / 120.0 * wheelDistance);
        updateFrame();
        event->accept();
        return;
    }
    QWidget::wheelEvent(event);
}

void Network3DView::keyPressEvent(QKeyEvent *event) {
    const int key = event->key();
    if (scene_ && (key == Qt::Key_Left || key == Qt::Key_Right ||
                   key == Qt::Key_Up || key == Qt::Key_Down)) {
        const double yaw = key == Qt::Key_Left ? -0.10 : key == Qt::Key_Right ? 0.10 : 0.0;
        const double pitch = key == Qt::Key_Up ? 0.10 : key == Qt::Key_Down ? -0.10 : 0.0;
        nn_3d_camera_look(&camera_, yaw, pitch);
        updateFrame();
        event->accept();
        return;
    }
    const int index = key == Qt::Key_W ? 0 : key == Qt::Key_S ? 1 : key == Qt::Key_A ? 2 : key == Qt::Key_D ? 3 :
                      key == Qt::Key_Q ? 4 : key == Qt::Key_E ? 5 : -1;
    if (index >= 0 && scene_ && !event->isAutoRepeat()) {
        keys_[index] = true;
        movementTimer_->start();
        event->accept();
        return;
    }
    QWidget::keyPressEvent(event);
}

void Network3DView::keyReleaseEvent(QKeyEvent *event) {
    const int key = event->key();
    const int index = key == Qt::Key_W ? 0 : key == Qt::Key_S ? 1 : key == Qt::Key_A ? 2 : key == Qt::Key_D ? 3 :
                      key == Qt::Key_Q ? 4 : key == Qt::Key_E ? 5 : -1;
    if (index >= 0 && !event->isAutoRepeat()) {
        keys_[index] = false;
        if (std::none_of(std::begin(keys_), std::end(keys_), [](bool pressed) { return pressed; })) movementTimer_->stop();
        event->accept();
        return;
    }
    QWidget::keyReleaseEvent(event);
}

void Network3DView::focusOutEvent(QFocusEvent *event) {
    std::fill(std::begin(keys_), std::end(keys_), false);
    movementTimer_->stop();
    rightDragging_ = false;
    QWidget::focusOutEvent(event);
}

void Network3DView::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    layout()->activate();
    updateFrame();
}

void Network3DView::refreshCameraAspect() {
    if (!scene_) return;
    const QRect canvas = sceneViewportRect();
    if (canvas.width() > 0 && canvas.height() > 0)
        nn_3d_camera_home(scene_, &camera_, qMax(0.1, canvas.width() / double(canvas.height())));
}

void Network3DView::tickMovement() {
    if (!scene_) return;
    const double forward = (keys_[0] ? 1.0 : 0.0) - (keys_[1] ? 1.0 : 0.0);
    const double right = (keys_[3] ? 1.0 : 0.0) - (keys_[2] ? 1.0 : 0.0);
    const double up = (keys_[5] ? 1.0 : 0.0) - (keys_[4] ? 1.0 : 0.0);
    const bool faster = QApplication::keyboardModifiers().testFlag(Qt::ShiftModifier);
    const double speed = faster ? 12.0 : 3.0;
    nn_3d_camera_move(&camera_, right * speed, up * speed, forward * speed);
    updateFrame();
}
