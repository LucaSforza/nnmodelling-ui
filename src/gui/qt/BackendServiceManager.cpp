#include "BackendServiceManager.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPointer>
#include <QTimer>
#include <QSaveFile>
#include <QStandardPaths>
#include <QTcpServer>
#include <cmath>
#include <utility>

namespace {
constexpr int WorkerCancelGraceMs = 60'000;
constexpr int WorkerShutdownGraceMs = WorkerCancelGraceMs + 5'000;

QJsonObject defaultProfile()
{
    QString jobRoot = QDir::homePath() + QStringLiteral("/.local/share/nnmodelling/jobs");
    return {{QStringLiteral("backend"), QJsonObject{
                {QStringLiteral("executor"), QStringLiteral("docker")},
                {QStringLiteral("job_root"), jobRoot},
                {QStringLiteral("container_runtime"), QStringLiteral("docker")},
                {QStringLiteral("worker_image"), QStringLiteral("nnmodelling-worker:local")}}},
            {QStringLiteral("slurm"), QJsonObject{
                {QStringLiteral("host"), QString()}, {QStringLiteral("root"), QString()},
                {QStringLiteral("image"), QString()}, {QStringLiteral("partition"), QString()},
                {QStringLiteral("cpus"), 2}, {QStringLiteral("memory"), QStringLiteral("4G")},
                {QStringLiteral("time"), QStringLiteral("00:30:00")},
                {QStringLiteral("gpus"), 0}, {QStringLiteral("ssh_executable"), QStringLiteral("ssh")}}}};
}

QString profilePath()
{
    QString path = qEnvironmentVariable("NNMODELLING_CONFIG");
    if (path.isEmpty()) {
        const QString xdg = qEnvironmentVariable("XDG_CONFIG_HOME");
        path = (xdg.isEmpty() ? QDir::homePath() + QStringLiteral("/.config") : xdg) +
            QStringLiteral("/nnmodelling/config.json");
    }
    if (path == QStringLiteral("~")) {
        path = QDir::homePath();
    } else if (path.startsWith(QStringLiteral("~/"))) {
        path = QDir::homePath() + path.mid(1);
    }
    return QDir::cleanPath(path);
}

bool validString(const QJsonValue &value)
{
    if (!value.isString()) return false;
    for (const QChar character : value.toString())
        if (character.unicode() < 32 || character.unicode() == 127) return false;
    return true;
}

bool validInteger(const QJsonValue &value, int minimum, int maximum)
{
    if (!value.isDouble()) return false;
    const double number = value.toDouble();
    return std::isfinite(number) && std::floor(number) == number && number >= minimum && number <= maximum;
}

bool validateSection(const QJsonObject &profile, const QString &section, const QStringList &keys,
                     QString *error)
{
    const QJsonValue value = profile.value(section);
    if (!value.isObject()) {
        if (error) *error = QObject::tr("Configuration profile contains an invalid object.");
        return false;
    }
    const QJsonObject fields = value.toObject();
    for (auto it = fields.begin(); it != fields.end(); ++it) {
        if (!keys.contains(it.key())) {
            if (error) *error = QObject::tr("Configuration profile contains unknown profile keys.");
            return false;
        }
    }
    return true;
}

bool validateProfile(QJsonObject *profile, QString *error)
{
    const QJsonObject defaults = defaultProfile();
    for (const QString &section : {QStringLiteral("backend"), QStringLiteral("slurm")}) {
        if (!profile->contains(section)) profile->insert(section, defaults.value(section));
    }
    if (!validateSection(*profile, QStringLiteral("backend"),
            {QStringLiteral("executor"), QStringLiteral("job_root"),
             QStringLiteral("container_runtime"), QStringLiteral("worker_image")}, error) ||
        !validateSection(*profile, QStringLiteral("slurm"),
            {QStringLiteral("host"), QStringLiteral("root"), QStringLiteral("image"),
             QStringLiteral("partition"), QStringLiteral("cpus"), QStringLiteral("memory"),
             QStringLiteral("time"), QStringLiteral("gpus"), QStringLiteral("ssh_executable")}, error))
        return false;

    QJsonObject backend = defaults.value(QStringLiteral("backend")).toObject();
    QJsonObject slurm = defaults.value(QStringLiteral("slurm")).toObject();
    const QJsonObject savedBackend = profile->value(QStringLiteral("backend")).toObject();
    const QJsonObject savedSlurm = profile->value(QStringLiteral("slurm")).toObject();
    for (auto it = savedBackend.begin(); it != savedBackend.end(); ++it) backend.insert(it.key(), it.value());
    for (auto it = savedSlurm.begin(); it != savedSlurm.end(); ++it) slurm.insert(it.key(), it.value());

    if (!validString(backend.value(QStringLiteral("executor"))) ||
        (backend.value(QStringLiteral("executor")).toString() != QStringLiteral("docker") &&
         backend.value(QStringLiteral("executor")).toString() != QStringLiteral("slurm"))) {
        if (error) *error = QObject::tr("Configuration profile has an invalid executor.");
        return false;
    }
    for (const QString &key : {QStringLiteral("job_root"), QStringLiteral("container_runtime"), QStringLiteral("worker_image")}) {
        if (!validString(backend.value(key))) {
            if (error) *error = QObject::tr("Configuration profile has an invalid string setting.");
            return false;
        }
    }
    for (const QString &key : {QStringLiteral("host"), QStringLiteral("root"), QStringLiteral("image"),
                               QStringLiteral("partition"), QStringLiteral("memory"), QStringLiteral("time"),
                               QStringLiteral("ssh_executable")}) {
        if (!validString(slurm.value(key))) {
            if (error) *error = QObject::tr("Configuration profile has an invalid string setting.");
            return false;
        }
    }
    if (!validInteger(slurm.value(QStringLiteral("cpus")), 1, 256) ||
        !validInteger(slurm.value(QStringLiteral("gpus")), 0, 8)) {
        if (error) *error = QObject::tr("Configuration profile has an invalid resource setting.");
        return false;
    }
    QString jobRoot = backend.value(QStringLiteral("job_root")).toString();
    if (jobRoot == QStringLiteral("~")) jobRoot = QDir::homePath();
    else if (jobRoot.startsWith(QStringLiteral("~/"))) jobRoot = QDir::homePath() + jobRoot.mid(1);
    backend.insert(QStringLiteral("job_root"), jobRoot);
    profile->insert(QStringLiteral("backend"), backend);
    profile->insert(QStringLiteral("slurm"), slurm);
    return true;
}

bool parseProfile(const QByteArray &bytes, QJsonObject *root, QString *error)
{
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        if (error) *error = QObject::tr("Configuration profile is malformed or has an invalid root object.");
        return false;
    }
    *root = document.object();
    return validateProfile(root, error);
}
}

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
    workerProcess_ = new QProcess(this);
    workerProcess_->setProcessChannelMode(QProcess::SeparateChannels);
    connect(workerProcess_, &QProcess::readyReadStandardError, this, [this] {
        // stderr may contain transport diagnostics; keep it out of the UI and logs.
        workerProcess_->readAllStandardError();
    });
    connect(workerProcess_, &QProcess::readyReadStandardOutput,
            this, &BackendServiceManager::consumeWorkerOutput);
    connect(workerProcess_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (error != QProcess::FailedToStart) return;
        const QString message = tr("Could not start worker preparation. Check that uv is installed.");
        workerFailureMessage_ = message;
        emit workerPreparationFinished(false, message);
    });
    connect(workerProcess_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus status) {
        workerOutput_ += workerProcess_->readAllStandardOutput();
        if (!workerOutput_.trimmed().isEmpty()) workerOutput_ += '\n';
        consumeWorkerOutput();
        if (workerCancelled_ || !workerFailureMessage_.isEmpty() || status != QProcess::NormalExit ||
            code != 0 || !workerSawSuccess_) {
            const QString message = !workerFailureMessage_.isEmpty() ? workerFailureMessage_
                : workerCancelled_ ? tr("Worker preparation cancelled.")
                : tr("Worker preparation failed (exit code %1).").arg(code);
            emit workerPreparationFinished(false, message);
            return;
        }
        QString path;
        QJsonObject profile;
        QString saveError;
        if (!loadUserConfig(&path, &profile, &saveError)) {
            emit workerPreparationFinished(false, tr("Worker prepared an image, but the configuration profile could not be loaded."));
            return;
        }
        if (path != preparationProfilePath_ ||
            profile.value(QStringLiteral("slurm")).toObject() != preparationSlurmProfile_) {
            emit workerPreparationFinished(false, tr("The Slurm profile changed while worker preparation was running; the image was not installed."));
            return;
        }
        if (workerCheckOnly_) {
            emit workerPreparationFinished(true, tr("Configured worker is ready."));
            return;
        }
        if (!QDir::isAbsolutePath(preparedImage_) || preparedImage_.split(QLatin1Char('/')).contains(QStringLiteral("..")) ||
            preparedImage_.contains(QChar::LineFeed) || preparedImage_.contains(QChar::CarriageReturn)) {
            emit workerPreparationFinished(false, tr("Worker preparation returned an invalid image path."));
            return;
        }
        QJsonObject slurm = profile.value(QStringLiteral("slurm")).toObject();
        slurm.insert(QStringLiteral("image"), preparedImage_);
        profile.insert(QStringLiteral("slurm"), slurm);
        if (!saveUserConfig(profile, &path, &saveError)) {
            emit workerPreparationFinished(false, tr("Worker prepared an image, but the configuration profile could not be updated."));
            return;
        }
        emit workerPreparationFinished(true, tr("Worker image prepared and installed."));
    });
}

void BackendServiceManager::consumeWorkerOutput()
{
        workerOutput_ += workerProcess_->readAllStandardOutput();
        if (workerOutput_.size() > 128 * 1024) {
            workerFailureMessage_ = tr("Worker preparation output exceeded its limit.");
            cancelWorkerPreparation();
            return;
        }
        qsizetype newline = -1;
        while ((newline = workerOutput_.indexOf('\n')) >= 0) {
            const QByteArray line = workerOutput_.left(newline).trimmed();
            workerOutput_.remove(0, newline + 1);
            if (line.isEmpty()) continue;
            if (line.size() > 16 * 1024) {
                workerFailureMessage_ = tr("Worker preparation emitted an oversized status event.");
                cancelWorkerPreparation();
                return;
            }
            QJsonParseError parseError;
            const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
            if (!document.isObject()) {
                workerFailureMessage_ = tr("Worker preparation emitted an invalid status event.");
                cancelWorkerPreparation();
                return;
            }
            const QJsonObject event = document.object();
            const QString phase = event.value(QStringLiteral("phase")).toString();
            QString message = event.value(QStringLiteral("message")).toString();
            const QString eventError = event.value(QStringLiteral("error")).toString();
            if (message.size() > 2000) message = message.left(2000);
            message.remove(QChar::Null);
            const qint64 current = event.value(QStringLiteral("current")).toVariant().toLongLong();
            const qint64 total = event.value(QStringLiteral("total")).toVariant().toLongLong();
            emit workerPreparationProgress(phase, message, current, total);
            if (phase == QStringLiteral("success")) {
                workerSawSuccess_ = true;
                preparedImage_ = event.value(QStringLiteral("image")).toString();
                if (!workerCheckOnly_ && (preparedImage_.isEmpty() || preparedImage_.contains(QChar::Null))) {
                    workerFailureMessage_ = tr("Worker preparation reported success without an image path.");
                    workerSawSuccess_ = false;
                }
            } else if (phase == QStringLiteral("error") || !eventError.isEmpty()) {
                workerFailureMessage_ = eventError.isEmpty()
                    ? (message.isEmpty() ? tr("Worker preparation failed.") : message)
                    : eventError.left(2000);
            }
        }
}

BackendServiceManager::~BackendServiceManager()
{
    cancelWorkerPreparation();
    if (workerProcess_ && workerProcess_->state() != QProcess::NotRunning &&
        !workerProcess_->waitForFinished(WorkerShutdownGraceMs)) {
        workerProcess_->kill();
        workerProcess_->waitForFinished(1000);
    }
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

bool BackendServiceManager::loadUserConfig(QString *path, QJsonObject *profile, QString *error) const
{
    const QString selectedPath = ::profilePath();
    if (path) *path = selectedPath;
    QFile file(selectedPath);
    if (!file.exists()) {
        if (qEnvironmentVariableIsSet("NNMODELLING_CONFIG") && !qEnvironmentVariable("NNMODELLING_CONFIG").isEmpty()) {
            if (error) *error = tr("The selected configuration profile does not exist.");
            return false;
        }
        if (profile) *profile = defaultProfile();
        return true;
    }
    if (!file.open(QIODevice::ReadOnly)) {
        if (error) *error = tr("Configuration profile is unreadable.");
        return false;
    }
    QJsonObject root;
    if (!parseProfile(file.readAll(), &root, error)) return false;
    if (profile) *profile = QJsonObject{{QStringLiteral("backend"), root.value(QStringLiteral("backend"))},
                                         {QStringLiteral("slurm"), root.value(QStringLiteral("slurm"))}};
    return true;
}

bool BackendServiceManager::saveUserConfig(const QJsonObject &profile, QString *path, QString *error) const
{
    const QString selectedPath = ::profilePath();
    if (path) *path = selectedPath;
    QFile existing(selectedPath);
    QJsonObject root;
    if (existing.exists()) {
        if (!existing.open(QIODevice::ReadOnly)) {
            if (error) *error = tr("Configuration profile is unreadable.");
            return false;
        }
        if (!parseProfile(existing.readAll(), &root, error)) return false;
    } else if (qEnvironmentVariableIsSet("NNMODELLING_CONFIG") && !qEnvironmentVariable("NNMODELLING_CONFIG").isEmpty()) {
        if (error) *error = tr("The selected configuration profile does not exist.");
        return false;
    }
    QJsonObject updated = profile;
    if (!validateProfile(&updated, error)) return false;
    root.insert(QStringLiteral("backend"), updated.value(QStringLiteral("backend")));
    root.insert(QStringLiteral("slurm"), updated.value(QStringLiteral("slurm")));
    const QFileInfo destination(selectedPath);
    if (!QDir().mkpath(destination.absolutePath())) {
        if (error) *error = tr("Configuration profile directory could not be created.");
        return false;
    }
    QSaveFile file(selectedPath);
    if (!file.open(QIODevice::WriteOnly)) {
        if (error) *error = tr("Configuration profile could not be saved.");
        return false;
    }
    const QByteArray bytes = QJsonDocument(root).toJson(QJsonDocument::Indented);
    if (file.write(bytes) != bytes.size() || !file.commit()) {
        if (error) *error = tr("Configuration profile could not be saved.");
        return false;
    }
    return true;
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

bool BackendServiceManager::prepareWorker(bool checkOnly, const QString &projectDirectory, QString *error)
{
    if (workerPreparationRunning()) {
        if (error) *error = tr("Worker preparation is already running.");
        return false;
    }
    if (!QFileInfo(sourceDirectory_ + QStringLiteral("/backend/prepare_worker.py")).isFile()) {
        if (error) *error = tr("Worker preparation helper is not available in this workspace yet.");
        return false;
    }
    const QString uv = QStandardPaths::findExecutable(QStringLiteral("uv"));
    if (uv.isEmpty()) {
        if (error) *error = tr("uv was not found on PATH.");
        return false;
    }
    workerOutput_.clear();
    preparedImage_.clear();
    workerFailureMessage_.clear();
    workerSawSuccess_ = false;
    workerCheckOnly_ = checkOnly;
    workerCancelled_ = false;
    QJsonObject profile;
    QString profileLoadError;
    if (!loadUserConfig(&preparationProfilePath_, &profile, &profileLoadError)) {
        if (error) *error = profileLoadError;
        return false;
    }
    preparationSlurmProfile_ = profile.value(QStringLiteral("slurm")).toObject();
    QStringList arguments{QStringLiteral("run"), QStringLiteral("--no-sync"), QStringLiteral("python"),
        QStringLiteral("-m"), QStringLiteral("backend.prepare_worker"),
        QStringLiteral("--profile"), ::profilePath()};
    if (!projectDirectory.isEmpty())
        arguments << QStringLiteral("--project") << QDir::cleanPath(projectDirectory);
    if (checkOnly) arguments << QStringLiteral("--check");
    workerProcess_->setWorkingDirectory(sourceDirectory_);
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    environment.remove(QStringLiteral("NNMODELLING_BEARER_TOKEN"));
    workerProcess_->setProcessEnvironment(environment);
    workerProcess_->setProgram(uv);
    workerProcess_->setArguments(arguments);
    workerProcess_->start();
    return true;
}

bool BackendServiceManager::workerPreparationRunning() const
{
    return workerProcess_ && workerProcess_->state() != QProcess::NotRunning;
}

void BackendServiceManager::cancelWorkerPreparation()
{
    if (!workerPreparationRunning() || workerCancelled_) return;
    workerCancelled_ = true;
    emit workerPreparationProgress(QStringLiteral("cancelling"),
        tr("Waiting for worker cleanup before stopping the helper…"), 0, 0);
    workerProcess_->terminate();
    QPointer<QProcess> process(workerProcess_);
    QTimer::singleShot(WorkerCancelGraceMs, this, [process] {
        if (process && process->state() != QProcess::NotRunning) process->kill();
    });
}
