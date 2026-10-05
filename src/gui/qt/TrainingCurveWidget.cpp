#include "TrainingCurveWidget.hpp"

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonValue>
#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
const QColor TrainingColor(QStringLiteral("#2677b8"));
const QColor ValidationColor(QStringLiteral("#d75252"));
const QColor GridColor(QStringLiteral("#dfe5eb"));
const QColor LabelColor(QStringLiteral("#455366"));

bool finiteNumber(const QJsonValue &value, double &number)
{
    if (!value.isDouble()) return false;
    number = value.toDouble();
    return std::isfinite(number);
}

QVector<QPointF> mapSeries(const QVector<QPointF> &points, const QRectF &plot,
                           double xMin, double xMax, double yMin, double yMax, bool logarithmic)
{
    QVector<QPointF> mapped;
    mapped.reserve(points.size());
    const double logMin = logarithmic ? std::log10(yMin) : yMin;
    const double logMax = logarithmic ? std::log10(yMax) : yMax;
    for (const QPointF &point : points) {
        if (logarithmic && point.y() <= 0.0) continue;
        const double y = logarithmic ? std::log10(point.y()) : point.y();
        const double xFraction = xMax == xMin ? 0.5 : (point.x() - xMin) / (xMax - xMin);
        const double yFraction = logMax == logMin ? 0.5 : (y - logMin) / (logMax - logMin);
        mapped.append(QPointF(plot.left() + xFraction * plot.width(), plot.bottom() - yFraction * plot.height()));
    }
    return mapped;
}

void drawSeries(QPainter &painter, const QVector<QPointF> &points, const QColor &color)
{
    if (points.isEmpty()) return;
    painter.setPen(QPen(color, 2.2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    if (points.size() > 1) {
        QPainterPath path(points.front());
        for (int i = 1; i < points.size(); ++i) path.lineTo(points.at(i));
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
    }
    painter.setBrush(color);
    for (const QPointF &point : points) painter.drawEllipse(point, 3.2, 3.2);
}
}

TrainingCurveWidget::TrainingCurveWidget(QWidget *parent) : QWidget(parent)
{
    setObjectName(QStringLiteral("trainingCurve"));
    setMinimumSize(420, 280);
    setAutoFillBackground(true);
    QPalette colors = palette();
    colors.setColor(QPalette::Window, Qt::white);
    setPalette(colors);
}

void TrainingCurveWidget::setMetrics(const QJsonObject &metrics)
{
    training_.clear();
    validation_.clear();
    invalidPoints_ = 0;
    const QJsonArray steps = metrics.value(QStringLiteral("steps")).toArray();
    const QJsonArray epochs = metrics.value(QStringLiteral("epochs")).toArray();
    stepAxis_ = !steps.isEmpty() || epochs.isEmpty();
    const QJsonArray points = stepAxis_ ? steps : epochs;
    double previousX = -std::numeric_limits<double>::infinity();
    for (const QJsonValue &value : points) {
        const QJsonObject point = value.toObject();
        double x = 0.0, training = 0.0, validation = 0.0;
        const QString xField = stepAxis_ ? QStringLiteral("step") : QStringLiteral("epoch");
        if (!finiteNumber(point.value(xField), x) || x <= 0.0 || std::floor(x) != x || x <= previousX ||
            !finiteNumber(point.value(QStringLiteral("training_loss")), training) ||
            !finiteNumber(point.value(QStringLiteral("validation_loss")), validation)) {
            ++invalidPoints_;
            continue;
        }
        previousX = x;
        training_.append(QPointF(x, training));
        validation_.append(QPointF(x, validation));
    }
    update();
}

void TrainingCurveWidget::setTrainingVisible(bool visible) { trainingVisible_ = visible; update(); }
void TrainingCurveWidget::setValidationVisible(bool visible) { validationVisible_ = visible; update(); }
void TrainingCurveWidget::setLogarithmic(bool logarithmic) { logarithmic_ = logarithmic; update(); }
QVector<QPointF> TrainingCurveWidget::trainingPoints() const { return training_; }
QVector<QPointF> TrainingCurveWidget::validationPoints() const { return validation_; }
bool TrainingCurveWidget::usesStepAxis() const { return stepAxis_; }
bool TrainingCurveWidget::isLogarithmic() const { return logarithmic_; }
QString TrainingCurveWidget::xAxisLabel() const { return stepAxis_ ? tr("Optimizer step") : tr("Epoch"); }

QString TrainingCurveWidget::dataNotice() const
{
    QString notice = invalidPoints_ == 0 ? QString() : tr("Ignored %1 invalid or non-finite metric point(s). ").arg(invalidPoints_);
    if (logarithmic_) {
        int nonPositive = 0;
        if (trainingVisible_)
            for (const QPointF &point : training_) if (point.y() <= 0.0) ++nonPositive;
        if (validationVisible_)
            for (const QPointF &point : validation_) if (point.y() <= 0.0) ++nonPositive;
        if (nonPositive) notice += tr("Log scale omits %1 non-positive value(s). ").arg(nonPositive);
    }
    return notice;
}

void TrainingCurveWidget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), Qt::white);
    painter.setFont(QFont(font().family(), 9));

    const QRectF plot = QRectF(rect()).adjusted(68, 24, -22, -52);
    if (plot.width() <= 0 || plot.height() <= 0) return;

    QVector<QPointF> active;
    if (trainingVisible_) active += training_;
    if (validationVisible_) active += validation_;
    QVector<double> positive;
    double xMin = std::numeric_limits<double>::infinity();
    double xMax = -std::numeric_limits<double>::infinity();
    double yMin = std::numeric_limits<double>::infinity();
    double yMax = -std::numeric_limits<double>::infinity();
    for (const QPointF &point : active) {
        xMin = qMin(xMin, point.x());
        xMax = qMax(xMax, point.x());
        if (logarithmic_ && point.y() <= 0.0) continue;
        yMin = qMin(yMin, point.y());
        yMax = qMax(yMax, point.y());
        if (point.y() > 0.0) positive.append(point.y());
    }
    const bool hasValues = std::isfinite(yMin) && std::isfinite(yMax);
    if (xMin == xMax) { xMin -= 0.5; xMax += 0.5; }
    if (!logarithmic_ && hasValues) {
        if (yMin >= 0.0) yMin = 0.0;
        if (yMax == yMin) yMax = yMin + qMax(1.0, std::abs(yMin) * 0.1);
        else yMax += (yMax - yMin) * 0.08;
    } else if (logarithmic_ && !positive.isEmpty()) {
        yMin = *std::min_element(positive.cbegin(), positive.cend());
        yMax = *std::max_element(positive.cbegin(), positive.cend());
        if (yMin == yMax) { yMin /= 10.0; yMax *= 10.0; }
    }

    painter.setPen(QPen(GridColor, 1));
    for (int i = 0; i <= 5; ++i) {
        const qreal y = plot.bottom() - plot.height() * i / 5.0;
        painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
        if (hasValues && (!logarithmic_ || !positive.isEmpty())) {
            double tick = yMin + (yMax - yMin) * i / 5.0;
            if (logarithmic_) tick = std::pow(10.0, std::log10(yMin) + (std::log10(yMax) - std::log10(yMin)) * i / 5.0);
            painter.setPen(LabelColor);
            painter.drawText(QRectF(0, y - 9, plot.left() - 9, 18), Qt::AlignRight | Qt::AlignVCenter,
                             QString::number(tick, 'g', 3));
            painter.setPen(QPen(GridColor, 1));
        }
    }
    for (int i = 0; i <= 5; ++i) {
        const qreal x = plot.left() + plot.width() * i / 5.0;
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
        if (std::isfinite(xMin) && std::isfinite(xMax)) {
            const double tick = xMin + (xMax - xMin) * i / 5.0;
            painter.setPen(LabelColor);
            painter.drawText(QRectF(x - 28, plot.bottom() + 5, 56, 18), Qt::AlignHCenter | Qt::AlignTop,
                             QString::number(tick, 'g', 3));
            painter.setPen(QPen(GridColor, 1));
        }
    }

    painter.setPen(QPen(QColor(QStringLiteral("#748196")), 1.2));
    painter.drawLine(plot.bottomLeft(), plot.topLeft());
    painter.drawLine(plot.bottomLeft(), plot.bottomRight());
    if (hasValues && (!logarithmic_ || !positive.isEmpty())) {
        if (trainingVisible_) drawSeries(painter, mapSeries(training_, plot, xMin, xMax, yMin, yMax, logarithmic_), TrainingColor);
        if (validationVisible_) drawSeries(painter, mapSeries(validation_, plot, xMin, xMax, yMin, yMax, logarithmic_), ValidationColor);
    }

    painter.setPen(LabelColor);
    painter.drawText(QRectF(plot.left(), height() - 25, plot.width(), 18), Qt::AlignHCenter, xAxisLabel());
    painter.save();
    painter.translate(15, plot.center().y());
    painter.rotate(-90);
    painter.drawText(QRectF(-plot.height() / 2, -9, plot.height(), 18), Qt::AlignHCenter, tr("Loss"));
    painter.restore();

    QString notice = dataNotice();
    if (!hasValues || (logarithmic_ && positive.isEmpty()))
        notice += tr("Waiting for finite %1 metrics.").arg(xAxisLabel().toLower());
    if (!notice.isEmpty()) {
        painter.setPen(LabelColor);
        painter.drawText(plot.adjusted(12, 10, -8, -8), Qt::AlignLeft | Qt::AlignTop | Qt::TextWordWrap, notice.trimmed());
    }
}
