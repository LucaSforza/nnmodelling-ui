#include "BackendServiceManager.hpp"

#include <QDir>
#include <QFileInfo>
#include <QHostAddress>
#include <QProcess>
#include <QProcessEnvironment>
#include <QStandardPaths>
#include <QTcpServer>
#include <utility>

BackendServiceManager::BackendServiceManager(QString sourceDirectory, QObject *parent)
    : QObject(parent), sourceDirectory_(QDir::cleanPath(std::move(sourceDirectory))),
      process_(new QProcess(this))
{
    connect(process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error == QProcess::FailedToStart)
            emit processError(tr("Could not start the local backend: %1").arg(process_->errorString()));
    });
    connect(process_, &QProcess::finished, this, [this](int code, QProcess::ExitStatus status) {
        port_ = 0;
        if (status == QProcess::CrashExit || code != 0)
            emit processError(tr("The managed backend stopped unexpectedly (exit code %1).").arg(code));
    });
}

BackendServiceManager::~BackendServiceManager()
{
    stop();
}

bool BackendServiceManager::isRunning() const
{
    return process_->state() != QProcess::NotRunning;
}

QUrl BackendServiceManager::endpoint() const
{
    if (port_ == 0) return {};
    return QUrl(QStringLiteral("http://127.0.0.1:%1").arg(port_));
}

QString BackendServiceManager::bearerToken() const
{
    return bearerToken_;
}

bool BackendServiceManager::start(const QStringList &environment, QString *error)
{
    const QFileInfo source(sourceDirectory_);
    if (!source.isDir() || !QFileInfo(sourceDirectory_ + QStringLiteral("/pyproject.toml")).isFile() ||
        !QFileInfo(sourceDirectory_ + QStringLiteral("/uv.lock")).isFile() ||
        !QFileInfo(sourceDirectory_ + QStringLiteral("/backend/app.py")).isFile()) {
        if (error) *error = tr("The local backend requires the repository source tree and uv workspace.");
        return false;
    }
    const QString uv = QStandardPaths::findExecutable(QStringLiteral("uv"));
    if (uv.isEmpty()) {
        if (error) *error = tr("uv was not found on PATH. Install uv and use the repository workspace before starting the local backend.");
        return false;
    }
    if (isRunning()) {
        if (error) *error = tr("The managed backend is already running.");
        return false;
    }
    QTcpServer portProbe;
    if (!portProbe.listen(QHostAddress::LocalHost, 0)) {
        if (error) *error = tr("Could not reserve a loopback port for the local backend.");
        return false;
    }
    port_ = portProbe.serverPort();
    portProbe.close();

    QProcessEnvironment processEnvironment = QProcessEnvironment::systemEnvironment();
    bearerToken_.clear();
    for (const QString &entry : environment) {
        const qsizetype separator = entry.indexOf(QLatin1Char('='));
        if (separator <= 0) continue;
        processEnvironment.insert(entry.left(separator), entry.mid(separator + 1));
        if (entry.left(separator) == QStringLiteral("NNMODELLING_BEARER_TOKEN"))
            bearerToken_ = entry.mid(separator + 1);
    }
    processEnvironment.insert(QStringLiteral("NNMODELLING_BACKEND_HOST"), QStringLiteral("127.0.0.1"));
    processEnvironment.insert(QStringLiteral("NNMODELLING_BACKEND_PORT"), QString::number(port_));
    process_->setProcessEnvironment(processEnvironment);
    process_->setProcessChannelMode(QProcess::ForwardedChannels);
    process_->setWorkingDirectory(sourceDirectory_);
    process_->setProgram(uv);
    process_->setArguments({QStringLiteral("run"), QStringLiteral("--package"),
        QStringLiteral("nnmodelling-backend"), QStringLiteral("--no-sync"),
        QStringLiteral("uvicorn"), QStringLiteral("backend.app:app"),
        QStringLiteral("--host"), QStringLiteral("127.0.0.1"),
        QStringLiteral("--port"), QString::number(port_)});
    process_->start();
    return true;
}

void BackendServiceManager::stop()
{
    if (!isRunning()) return;
    process_->terminate();
    if (!process_->waitForFinished(4000)) {
        process_->kill();
        process_->waitForFinished(2000);
    }
    port_ = 0;
    bearerToken_.clear();
}
