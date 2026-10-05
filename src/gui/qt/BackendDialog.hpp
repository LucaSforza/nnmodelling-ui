#ifndef NN_GUI_BACKEND_DIALOG_HPP
#define NN_GUI_BACKEND_DIALOG_HPP

#include <QDialog>
#include <QUrl>
#include <functional>

class QLineEdit;
class QLabel;
class QTreeWidget;
class QPlainTextEdit;
class QNetworkAccessManager;
class QComboBox;
class QCheckBox;
class TrainingCurveWidget;
class NNApplication;

class BackendDialog final : public QDialog {
    Q_OBJECT
public:
    BackendDialog(NNApplication *application, std::function<bool()> saveProject,
                  std::function<bool(const QString &)> openProject, QWidget *parent = nullptr);

private:
    void request(const QString &path, const QByteArray &method = QByteArrayLiteral("GET"),
                 const QByteArray &body = {});
    void connectBackend();
    void refreshJobs();
    void submitJob();
    void showJob(const QString &id);
    void download(const QString &kind);
    void restoreSnapshot();
    void setMessage(const QString &message, bool error = false);
    QUrl endpointUrl(const QString &path) const;

    NNApplication *application_ = nullptr;
    std::function<bool()> saveProject_;
    std::function<bool(const QString &)> openProject_;
    QNetworkAccessManager *network_ = nullptr;
    QLineEdit *endpoint_ = nullptr;
    QLineEdit *token_ = nullptr;
    QLineEdit *epochs_ = nullptr;
    QLineEdit *batchSize_ = nullptr;
    QLineEdit *learningRate_ = nullptr;
    QLineEdit *seed_ = nullptr;
    QLineEdit *publishEverySteps_ = nullptr;
    QLabel *status_ = nullptr;
    QLabel *finalTestLoss_ = nullptr;
    QTreeWidget *jobs_ = nullptr;
    QPlainTextEdit *metrics_ = nullptr;
    TrainingCurveWidget *curve_ = nullptr;
    QCheckBox *trainingCurveVisible_ = nullptr;
    QCheckBox *validationCurveVisible_ = nullptr;
    QComboBox *curveScale_ = nullptr;
    QString selectedJob_;
    QString jobsEndpointInFlight_;
    bool jobsRequestInFlight_ = false;
};

#endif
