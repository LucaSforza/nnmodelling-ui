#ifndef NN_GUI_BACKEND_SERVICE_MANAGER_HPP
#define NN_GUI_BACKEND_SERVICE_MANAGER_HPP

#include <QObject>
#include <QUrl>
#include <QString>
#include <QStringList>

class QProcess;

class BackendServiceManager final : public QObject {
    Q_OBJECT
public:
    explicit BackendServiceManager(QString sourceDirectory, QObject *parent = nullptr);
    ~BackendServiceManager() override;

    bool isRunning() const;
    QUrl endpoint() const;
    QString bearerToken() const;
    bool start(const QStringList &environment, QString *error = nullptr);
    void stop();

signals:
    void processError(const QString &message);

private:
    QString sourceDirectory_;
    QProcess *process_ = nullptr;
    quint16 port_ = 0;
    QString bearerToken_;
};

#endif
