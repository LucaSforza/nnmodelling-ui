#ifndef NN_GUI_BACKEND_SERVICE_MANAGER_HPP
#define NN_GUI_BACKEND_SERVICE_MANAGER_HPP

#include <QObject>
#include <QUrl>
#include <QString>
#include <QStringList>
#include <QJsonObject>
#include <QByteArray>

class QProcess;

class BackendServiceManager final : public QObject {
    Q_OBJECT
public:
    explicit BackendServiceManager(QString sourceDirectory, QObject *parent = nullptr);
    ~BackendServiceManager() override;

    bool isRunning() const;
    QUrl endpoint() const;
    QString bearerToken() const;
    bool loadUserConfig(QString *path, QJsonObject *profile, QString *error = nullptr) const;
    bool saveUserConfig(const QJsonObject &profile, QString *path, QString *error = nullptr) const;
    bool start(const QStringList &environment, QString *error = nullptr);
    void stop();
    bool prepareWorker(bool checkOnly, const QString &projectDirectory = QString(), QString *error = nullptr);
    bool workerPreparationRunning() const;
    void cancelWorkerPreparation();

signals:
    void processError(const QString &message);
    void workerPreparationProgress(const QString &phase, const QString &message,
                                  qint64 current, qint64 total);
    void workerPreparationFinished(bool success, const QString &message);

private:
    void consumeWorkerOutput();
    QString sourceDirectory_;
    QProcess *process_ = nullptr;
    QProcess *workerProcess_ = nullptr;
    QByteArray workerOutput_;
    QString preparedImage_;
    QString workerFailureMessage_;
    QString preparationProfilePath_;
    QJsonObject preparationSlurmProfile_;
    bool workerSawSuccess_ = false;
    bool workerCheckOnly_ = false;
    bool workerCancelled_ = false;
    quint16 port_ = 0;
    QString bearerToken_;
};

#endif
