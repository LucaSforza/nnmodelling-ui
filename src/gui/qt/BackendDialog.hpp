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
class QPushButton;
class TrainingCurveWidget;
class NNApplication;
class BackendServiceManager;

class BackendDialog final : public QDialog {
    Q_OBJECT
public:
    BackendDialog(NNApplication *application, std::function<bool()> saveProject,
                  std::function<bool(const QString &)> openProject, QWidget *parent = nullptr,
                  BackendServiceManager *serviceManager = nullptr);

private:
    void request(const QString &path, const QByteArray &method = QByteArrayLiteral("GET"),
                 const QByteArray &body = {});
    void connectBackend();
    void refreshJobs();
    void submitJob();
    void showJob(const QString &id);
    void download(const QString &kind);
    void restoreSnapshot();
    void startManagedBackend();
    void checkJobsBeforeRestart();
    void saveLauncherSettings();
    void setMessage(const QString &message, bool error = false);
    QUrl endpointUrl(const QString &path) const;

    NNApplication *application_ = nullptr;
    std::function<bool()> saveProject_;
    std::function<bool(const QString &)> openProject_;
    BackendServiceManager *serviceManager_ = nullptr;
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
    QComboBox *executor_ = nullptr;
    QLineEdit *jobRoot_ = nullptr;
    QLineEdit *dockerRuntime_ = nullptr;
    QLineEdit *dockerImage_ = nullptr;
    QLineEdit *slurmHost_ = nullptr;
    QLineEdit *slurmRoot_ = nullptr;
    QLineEdit *slurmImage_ = nullptr;
    QLineEdit *slurmPartition_ = nullptr;
    QLineEdit *slurmCpus_ = nullptr;
    QLineEdit *slurmMemory_ = nullptr;
    QLineEdit *slurmTime_ = nullptr;
    QPushButton *managedBackendButton_ = nullptr;
    QString selectedJob_;
    QString jobsEndpointInFlight_;
    bool jobsRequestInFlight_ = false;
};

#endif
