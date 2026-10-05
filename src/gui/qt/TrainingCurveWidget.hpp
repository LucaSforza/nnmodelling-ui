#ifndef NN_GUI_TRAINING_CURVE_WIDGET_HPP
#define NN_GUI_TRAINING_CURVE_WIDGET_HPP

#include <QPointF>
#include <QVector>
#include <QWidget>

class QJsonObject;

class TrainingCurveWidget final : public QWidget {
    Q_OBJECT
public:
    explicit TrainingCurveWidget(QWidget *parent = nullptr);

    void setMetrics(const QJsonObject &metrics);
    void setTrainingVisible(bool visible);
    void setValidationVisible(bool visible);
    void setLogarithmic(bool logarithmic);

    QVector<QPointF> trainingPoints() const;
    QVector<QPointF> validationPoints() const;
    bool usesStepAxis() const;
    bool isLogarithmic() const;
    QString xAxisLabel() const;
    QString dataNotice() const;

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QVector<QPointF> training_;
    QVector<QPointF> validation_;
    bool trainingVisible_ = true;
    bool validationVisible_ = true;
    bool logarithmic_ = false;
    bool stepAxis_ = false;
    int invalidPoints_ = 0;
};

#endif
