#include "BackendDialog.hpp"

#include "application/application.h"
#include "project/project.h"

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSet>
#include <QTemporaryFile>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <cmath>

namespace {
constexpr qint64 MaxModelBytes = 8 * 1024 * 1024;
constexpr qint64 MaxResourceBytes = 256LL * 1024 * 1024;
constexpr qint64 MaxResourceFileBytes = 128LL * 1024 * 1024;
constexpr qint64 MaxJsonReplyBytes = 512LL * 1024 * 1024;
constexpr qint64 MaxArtifactBytes = 1024LL * 1024 * 1024;
constexpr int MaxResourceFiles = 10000;
constexpr int RequestTimeoutMs = 15000;

bool safeRelativePath(const QString &path)
{
    if (path.isEmpty() || path.startsWith('/') || path.contains('\\') || path.contains(':') || path.contains(QChar::Null)) return false;
    const QStringList parts = path.split('/', Qt::KeepEmptyParts);
    for (const QString &part : parts)
        if (part.isEmpty() || part == QStringLiteral(".") || part == QStringLiteral("..")) return false;
    return true;
}

bool excludedResourcePath(const QString &relative)
{
    const QStringList parts = relative.split('/');
    for (const QString &part : parts)
        if (part == QStringLiteral(".venv") || part == QStringLiteral("venv") ||
            part == QStringLiteral("env") || part == QStringLiteral("__pycache__") ||
            part == QStringLiteral(".git") || part == QStringLiteral("build") ||
            part == QStringLiteral("dist") || part == QStringLiteral("target") ||
            part == QStringLiteral(".pytest_cache") || part == QStringLiteral(".nnmodelling-generated")) return true;
    const QString name = parts.isEmpty() ? QString() : parts.back();
    return name == QStringLiteral(".DS_Store") || name.endsWith(QStringLiteral(".pyc")) ||
        name.endsWith(QStringLiteral(".pyo")) || name.endsWith(QLatin1Char('~')) ||
        name.contains(QStringLiteral(".generated."));
}

bool gatherFile(const QString &base, const QString &relative, QJsonObject &files,
                QSet<QString> &included, qint64 &total, QString &error)
{
    if (included.contains(relative) || relative == QStringLiteral("model.json") || excludedResourcePath(relative)) return true;
    if (!safeRelativePath(relative)) { error = QObject::tr("Project contains an unsafe resource path."); return false; }
    const QFileInfo info(QDir(base).filePath(relative));
    if (info.isSymLink() || !info.isFile()) return true;
    if (info.size() < 0 || info.size() > MaxResourceFileBytes) {
        error = QObject::tr("Project resource exceeds the 128 MiB per-file upload limit: %1").arg(relative);
        return false;
    }
    if (total > MaxResourceBytes - info.size() || included.size() >= MaxResourceFiles) {
        error = QObject::tr("Project resources exceed the 256 MiB total upload or 10,000-file limit.");
        return false;
    }
    QFile file(info.absoluteFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        error = QObject::tr("Cannot read project resource %1: %2").arg(relative, file.errorString());
        return false;
    }
    const QByteArray bytes = file.read(qMin(MaxResourceFileBytes, MaxResourceBytes - total) + 1);
    if (bytes.size() != info.size() || bytes.size() > MaxResourceBytes - total) {
        error = QObject::tr("Project resource changed or exceeded its upload limit: %1").arg(relative);
        return false;
    }
    total += bytes.size();
    included.insert(relative);
    files.insert(relative, QString::fromLatin1(bytes.toBase64()));
    return true;
}

bool gatherFiles(const QString &base, const QString &relativeDirectory, QJsonObject &files,
                 QSet<QString> &included, qint64 &total, QString &error)
{
    if (!safeRelativePath(relativeDirectory) || relativeDirectory.count(QLatin1Char('/')) > 128) {
        error = QObject::tr("Project resource path or directory nesting exceeds the upload limit.");
        return false;
    }
    if (excludedResourcePath(relativeDirectory)) return true;
    const QFileInfo root(QDir(base).filePath(relativeDirectory));
    if (root.isSymLink()) return true;
    if (root.isFile()) return gatherFile(base, relativeDirectory, files, included, total, error);
    QDir dir(root.absoluteFilePath());
    const QFileInfoList entries = dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden,
                                                     QDir::Name);
    for (const QFileInfo &info : entries) {
        const QString relative = relativeDirectory + QLatin1Char('/') + info.fileName();
        if (!safeRelativePath(relative) || excludedResourcePath(relative)) continue;
        if (info.isSymLink()) continue;
        if (info.isDir()) {
            if (!gatherFiles(base, relative, files, included, total, error)) return false;
            continue;
        }
        if (!gatherFile(base, relative, files, included, total, error)) return false;
    }
    return true;
}

QString formatJob(const QJsonObject &job)
{
    const QJsonObject metrics = job.value(QStringLiteral("metrics")).toObject();
    QStringList lines;
    lines << QObject::tr("Status: %1").arg(job.value(QStringLiteral("status")).toString());
    if (!job.value(QStringLiteral("error")).isNull() && !job.value(QStringLiteral("error")).toString().isEmpty())
        lines << QObject::tr("Error: %1").arg(job.value(QStringLiteral("error")).toString());
    lines << QObject::tr("Epoch metrics:");
    const QJsonArray epochs = metrics.value(QStringLiteral("epochs")).toArray();
    for (const QJsonValue &value : epochs) {
        const QJsonObject row = value.toObject();
        lines << QObject::tr("Epoch %1 — train %2, validation %3")
            .arg(row.value(QStringLiteral("epoch")).toInt())
            .arg(row.value(QStringLiteral("training_loss")).toVariant().toString(),
                 row.value(QStringLiteral("validation_loss")).toVariant().toString());
    }
    const QJsonValue testLoss = metrics.value(QStringLiteral("test_loss"));
    lines << QObject::tr("Final test loss: %1").arg(testLoss.isNull() ? QObject::tr("pending") : testLoss.toVariant().toString());
    return lines.join(QLatin1Char('\n'));
}

QString backendError(const QByteArray &body, int status, const QString &fallback)
{
    const QJsonDocument document = QJsonDocument::fromJson(body);
    QString detail = document.object().value(QStringLiteral("detail")).toString();
    if (detail.isEmpty()) detail = QString::fromUtf8(body.left(2048)).trimmed();
    if (detail.isEmpty()) detail = fallback;
    return status > 0 ? QObject::tr("Backend request failed (%1): %2").arg(status).arg(detail)
                      : QObject::tr("Backend request failed: %1").arg(detail);
}
}

BackendDialog::BackendDialog(NNApplication *application, std::function<bool()> saveProject,
                             std::function<bool(const QString &)> openProject, QWidget *parent)
    : QDialog(parent), application_(application), saveProject_(std::move(saveProject)),
      openProject_(std::move(openProject)), network_(new QNetworkAccessManager(this))
{
    setWindowTitle(tr("Training backend"));
    resize(740, 620);
    auto *layout = new QVBoxLayout(this);
    auto *connection = new QFormLayout;
    endpoint_ = new QLineEdit(QStringLiteral("http://127.0.0.1:8765"), this);
    endpoint_->setObjectName(QStringLiteral("backendEndpoint"));
    token_ = new QLineEdit(this);
    token_->setObjectName(QStringLiteral("backendToken"));
    token_->setEchoMode(QLineEdit::Password);
    connection->addRow(tr("Endpoint"), endpoint_);
    connection->addRow(tr("Bearer token"), token_);
    layout->addLayout(connection);
    auto *connectionButtons = new QHBoxLayout;
    auto *connectButton = new QPushButton(tr("Connect / check health"), this);
    connectButton->setObjectName(QStringLiteral("backendConnect"));
    auto *refreshButton = new QPushButton(tr("Refresh jobs"), this);
    refreshButton->setObjectName(QStringLiteral("backendRefresh"));
    connectionButtons->addWidget(connectButton);
    connectionButtons->addWidget(refreshButton);
    connectionButtons->addStretch(1);
    layout->addLayout(connectionButtons);
    status_ = new QLabel(tr("Not connected"), this);
    status_->setObjectName(QStringLiteral("backendStatus"));
    status_->setWordWrap(true);
    layout->addWidget(status_);

    auto *training = new QFormLayout;
    epochs_ = new QLineEdit(QStringLiteral("10"), this);
    batchSize_ = new QLineEdit(QStringLiteral("32"), this);
    learningRate_ = new QLineEdit(QStringLiteral("0.001"), this);
    seed_ = new QLineEdit(QStringLiteral("0"), this);
    epochs_->setObjectName(QStringLiteral("trainingEpochs"));
    batchSize_->setObjectName(QStringLiteral("trainingBatchSize"));
    learningRate_->setObjectName(QStringLiteral("trainingLearningRate"));
    seed_->setObjectName(QStringLiteral("trainingSeed"));
    training->addRow(tr("Epochs"), epochs_);
    training->addRow(tr("Batch size"), batchSize_);
    training->addRow(tr("Learning rate"), learningRate_);
    training->addRow(tr("Seed"), seed_);
    layout->addLayout(training);
    auto *submitButton = new QPushButton(tr("Save project and submit"), this);
    submitButton->setObjectName(QStringLiteral("backendSubmit"));
    layout->addWidget(submitButton);

    jobs_ = new QTreeWidget(this);
    jobs_->setObjectName(QStringLiteral("backendJobs"));
    jobs_->setColumnCount(3);
    jobs_->setHeaderLabels({tr("Job"), tr("Status"), tr("Created")});
    jobs_->header()->setStretchLastSection(true);
    layout->addWidget(jobs_, 1);
    metrics_ = new QPlainTextEdit(this);
    metrics_->setObjectName(QStringLiteral("backendMetrics"));
    metrics_->setReadOnly(true);
    metrics_->setMaximumHeight(130);
    layout->addWidget(metrics_);
    auto *actions = new QHBoxLayout;
    for (const auto &entry : {qMakePair(tr("Download weights"), QStringLiteral("weights")),
                               qMakePair(tr("Download wheel"), QStringLiteral("wheel")),
                               qMakePair(tr("Restore snapshot…"), QStringLiteral("restore")),
                               qMakePair(tr("Cancel job"), QStringLiteral("cancel"))}) {
        auto *button = new QPushButton(entry.first, this);
        button->setObjectName(QStringLiteral("backend_") + entry.second);
        actions->addWidget(button);
        connect(button, &QPushButton::clicked, this, [this, key = entry.second] {
            if (key == QStringLiteral("restore")) restoreSnapshot();
            else if (key == QStringLiteral("cancel")) {
                if (!selectedJob_.isEmpty()) request(QStringLiteral("/v1/jobs/%1/cancel").arg(selectedJob_), QByteArrayLiteral("POST"), QByteArrayLiteral("{}"));
            } else download(key);
        });
    }
    auto *closeButton = new QPushButton(tr("Close"), this);
    actions->addWidget(closeButton);
    layout->addLayout(actions);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(connectButton, &QPushButton::clicked, this, &BackendDialog::connectBackend);
    connect(refreshButton, &QPushButton::clicked, this, &BackendDialog::refreshJobs);
    connect(submitButton, &QPushButton::clicked, this, &BackendDialog::submitJob);
    connect(jobs_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem *current) {
        if (!current) return;
        selectedJob_ = current->data(0, Qt::UserRole).toString();
        showJob(selectedJob_);
    });
    auto *poll = new QTimer(this);
    poll->setInterval(2500);
    connect(poll, &QTimer::timeout, this, [this] { if (!endpoint_->text().isEmpty()) refreshJobs(); });
    poll->start();
}

QUrl BackendDialog::endpointUrl(const QString &path) const
{
    QUrl base(endpoint_->text().trimmed());
    if (!base.isValid() || (base.scheme() != QStringLiteral("http") && base.scheme() != QStringLiteral("https"))) return {};
    QString root = base.path();
    while (root.endsWith('/')) root.chop(1);
    base.setPath(root + path);
    base.setQuery({});
    base.setFragment({});
    return base;
}

void BackendDialog::setMessage(const QString &message, bool error)
{
    status_->setText(message);
    status_->setStyleSheet(error ? QStringLiteral("color: #a22; font-weight: 600;") : QString());
}

void BackendDialog::request(const QString &path, const QByteArray &method, const QByteArray &body)
{
    if (path == QStringLiteral("/v1/jobs") && jobsRequestInFlight_) return;
    const QUrl url = endpointUrl(path);
    if (!url.isValid()) { setMessage(tr("Enter a valid HTTP or HTTPS endpoint."), true); return; }
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    const QString token = token_->text();
    if (!token.isEmpty()) request.setRawHeader("Authorization", QByteArrayLiteral("Bearer ") + token.toUtf8());
    QNetworkReply *reply = method == QByteArrayLiteral("POST")
        ? network_->post(request, body) : network_->get(request);
    if (path == QStringLiteral("/v1/jobs")) jobsRequestInFlight_ = true;
    reply->setProperty("endpoint", endpoint_->text().trimmed());
    auto *timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, reply, [reply] { reply->abort(); });
    timeout->start(RequestTimeoutMs);
    connect(reply, &QNetworkReply::downloadProgress, this, [reply](qint64 received, qint64) {
        if (received > MaxJsonReplyBytes) reply->abort();
    });
    connect(reply, &QNetworkReply::finished, this, [this, reply, path, method] {
        const QByteArray bytes = reply->readAll();
        const auto networkError = reply->error();
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        if (path == QStringLiteral("/v1/jobs")) jobsRequestInFlight_ = false;
        const bool staleEndpoint = reply->property("endpoint").toString() != endpoint_->text().trimmed();
        reply->deleteLater();
        if (staleEndpoint) return;
        if (bytes.size() > MaxJsonReplyBytes) { setMessage(tr("Backend JSON response exceeded the 512 MiB limit."), true); return; }
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
        if (networkError != QNetworkReply::NoError || status < 200 || status >= 300) {
            setMessage(backendError(bytes, status, reply->errorString()), true);
            return;
        }
        if (!document.isObject() && path != QStringLiteral("/health")) {
            setMessage(tr("Backend returned invalid JSON: %1").arg(parseError.errorString()), true);
            return;
        }
        setMessage(tr("Backend request completed."));
        const QJsonObject object = document.object();
        if (path == QStringLiteral("/health")) {
            setMessage(tr("Connected: %1").arg(QString::fromUtf8(bytes.left(2048))));
            refreshJobs();
        } else if (path == QStringLiteral("/v1/jobs")) {
            const QString oldSelection = selectedJob_;
            const QSignalBlocker blocker(jobs_);
            jobs_->clear();
            for (const QJsonValue &entry : object.value(QStringLiteral("jobs")).toArray()) {
                const QJsonObject job = entry.toObject();
                const QString id = job.value(QStringLiteral("id")).toString();
                auto *item = new QTreeWidgetItem(jobs_, {id, job.value(QStringLiteral("status")).toString(),
                                                          job.value(QStringLiteral("created_at")).toString()});
                item->setData(0, Qt::UserRole, id);
                if (id == oldSelection) jobs_->setCurrentItem(item);
            }
            if (!selectedJob_.isEmpty()) showJob(selectedJob_);
        } else if (path.contains(QStringLiteral("/cancel"))) {
            showJob(selectedJob_);
            refreshJobs();
        } else if (path.endsWith(QStringLiteral("/snapshot"))) {
            // Handled by restoreSnapshot through a one-shot callback below.
        } else if (method == QByteArrayLiteral("POST") && object.contains(QStringLiteral("id"))) {
            selectedJob_ = object.value(QStringLiteral("id")).toString();
            refreshJobs();
            showJob(selectedJob_);
        } else if (path == QStringLiteral("/v1/jobs/%1").arg(selectedJob_)) {
            const QString text = formatJob(object);
            if (metrics_->toPlainText() != text) {
                QScrollBar *scroll = metrics_->verticalScrollBar();
                const bool followBottom = scroll->value() >= scroll->maximum();
                const int position = scroll->value();
                metrics_->setPlainText(text);
                scroll->setValue(followBottom ? scroll->maximum() : qMin(position, scroll->maximum()));
            }
        }
    });
}

void BackendDialog::connectBackend()
{
    request(QStringLiteral("/health"));
}

void BackendDialog::refreshJobs()
{
    request(QStringLiteral("/v1/jobs"));
}

void BackendDialog::submitJob()
{
    bool epochsOk = false, batchOk = false, seedOk = false, lrOk = false;
    const int epochs = epochs_->text().toInt(&epochsOk);
    const int batch = batchSize_->text().toInt(&batchOk);
    const qint64 seed = seed_->text().toLongLong(&seedOk);
    const double learningRate = learningRate_->text().toDouble(&lrOk);
    if (!epochsOk || epochs < 1 || epochs > 10000 || !batchOk || batch < 1 || batch > 4096 ||
        !seedOk || seed < -2147483648LL || seed > 4294967295LL || !lrOk ||
        !std::isfinite(learningRate) || learningRate <= 0.0 || learningRate > 1.0) {
        setMessage(tr("Training settings require positive epochs, batch size, learning rate and a valid seed."), true);
        return;
    }
    if (!application_ || !nn_app_project(application_) || !saveProject_ || !saveProject_()) {
        setMessage(tr("Save the current project before submitting a training job."), true);
        return;
    }
    const NNProject *project = nn_app_project(application_);
    QFile model(QDir(QString::fromUtf8(nn_project_directory(project))).filePath(QStringLiteral("model.json")));
    if (!model.open(QIODevice::ReadOnly) || model.size() > MaxModelBytes) {
        setMessage(tr("Cannot read the saved model or it exceeds the upload limit."), true);
        return;
    }
    const QByteArray modelBytes = model.read(MaxModelBytes + 1);
    if (modelBytes.size() > MaxModelBytes || modelBytes.size() != model.size()) {
        setMessage(tr("Saved model changed or exceeded the upload limit."), true);
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument modelDocument = QJsonDocument::fromJson(modelBytes, &parseError);
    if (!modelDocument.isObject()) { setMessage(tr("Saved model JSON is invalid: %1").arg(parseError.errorString()), true); return; }
    QJsonObject files;
    QSet<QString> included;
    qint64 total = 0;
    QString gatherError;
    const QString root = QString::fromUtf8(nn_project_directory(project));
    const QJsonObject manifest = modelDocument.object().value(QStringLiteral("manifest")).toObject();
    for (const QString &listName : {QStringLiteral("customPackages"), QStringLiteral("customDatasets")}) {
        for (const QJsonValue &reference : manifest.value(listName).toArray()) {
            const QString path = reference.toObject().value(QStringLiteral("path")).toString();
            if (path.isEmpty() || !gatherFiles(root, path, files, included, total, gatherError)) {
                setMessage(gatherError.isEmpty() ? tr("Project resource reference has no safe path.") : gatherError, true);
                return;
            }
        }
    }
    for (const QString &folder : {QStringLiteral("data"), QStringLiteral("resources")})
        if (!gatherFiles(root, folder, files, included, total, gatherError)) { setMessage(gatherError, true); return; }
    // Explicit inference assets may live outside the conventional resource folders.
    const QJsonObject uploaded = files;
    for (auto it = uploaded.begin(); it != uploaded.end(); ++it) {
        if (!it.key().endsWith(QStringLiteral("/manifest.json"))) continue;
        const QJsonDocument resourceManifest = QJsonDocument::fromJson(
            QByteArray::fromBase64(it.value().toString().toLatin1()));
        const QJsonArray assets = resourceManifest.object().value(QStringLiteral("inferenceAssets")).toArray();
        for (const QJsonValue &asset : assets) {
            const QString path = asset.toString();
            if (path.isEmpty() || !safeRelativePath(path)) {
                setMessage(tr("Resource manifest contains an unsafe inference asset path."), true);
                return;
            }
            if (excludedResourcePath(path)) {
                setMessage(tr("Inference assets cannot be inside an excluded generated or environment directory."), true);
                return;
            }
            const QFileInfo assetInfo(QDir(root).filePath(path));
            if (assetInfo.isSymLink() || !assetInfo.isFile() ||
                !gatherFiles(root, path, files, included, total, gatherError)) {
                setMessage(gatherError.isEmpty() ? tr("Resource manifest contains an unsafe inference asset path.") : gatherError, true);
                return;
            }
        }
    }
    const QJsonObject payload{
        {QStringLiteral("project"), modelDocument.object()},
        {QStringLiteral("files"), files},
        {QStringLiteral("training"), QJsonObject{{QStringLiteral("epochs"), epochs},
            {QStringLiteral("batch_size"), batch}, {QStringLiteral("learning_rate"), learningRate},
            {QStringLiteral("seed"), static_cast<double>(seed)}}}
    };
    setMessage(tr("Submitting saved project snapshot…"));
    const QByteArray body = QJsonDocument(payload).toJson(QJsonDocument::Compact);
    const qint64 maxRequestBytes = MaxResourceBytes * 4 / 3 + MaxModelBytes + 1024 * 1024;
    if (body.size() > maxRequestBytes) { setMessage(tr("Encoded upload exceeds the backend request limit."), true); return; }
    request(QStringLiteral("/v1/jobs"), QByteArrayLiteral("POST"), body);
}

void BackendDialog::showJob(const QString &id)
{
    if (id.isEmpty()) return;
    request(QStringLiteral("/v1/jobs/%1").arg(id));
}

void BackendDialog::download(const QString &kind)
{
    if (selectedJob_.isEmpty()) { setMessage(tr("Select a job first."), true); return; }
    const QString apiPath = QStringLiteral("/v1/jobs/%1/%2").arg(selectedJob_, kind);
    const QUrl url = endpointUrl(apiPath);
    QNetworkRequest request(url);
    const QString token = token_->text();
    if (!token.isEmpty()) request.setRawHeader("Authorization", QByteArrayLiteral("Bearer ") + token.toUtf8());
    QNetworkReply *reply = network_->get(request);
    reply->setProperty("endpoint", endpoint_->text().trimmed());
    reply->setProperty("token", token);
    auto *timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, reply, [reply] { reply->abort(); });
    timeout->start(RequestTimeoutMs);
    auto *file = new QTemporaryFile(QDir::tempPath() + QStringLiteral("/nnmodelling-download-XXXXXX"), reply);
    if (!file->open()) {
        setMessage(tr("Cannot create download file: %1").arg(file->errorString()), true);
        reply->abort();
    }
    reply->setProperty("downloadBytes", QVariant::fromValue<qint64>(0));
    connect(reply, &QNetworkReply::readyRead, this, [this, reply, file] {
        const QByteArray chunk = reply->readAll();
        const qint64 total = reply->property("downloadBytes").toLongLong() + chunk.size();
        if (total > MaxArtifactBytes || file->write(chunk) != chunk.size()) {
            reply->setProperty("downloadFailed", true);
            if (total > MaxArtifactBytes) reply->setProperty("downloadTooLarge", true);
            reply->abort();
            return;
        }
        reply->setProperty("downloadBytes", QVariant::fromValue(total));
    });
    connect(reply, &QNetworkReply::finished, this, [this, reply, file, kind] {
        const QString error = reply->errorString();
        if (file && reply->bytesAvailable()) {
            const QByteArray chunk = reply->readAll();
            if (reply->property("downloadBytes").toLongLong() + chunk.size() > MaxArtifactBytes) {
                reply->setProperty("downloadFailed", true);
                reply->setProperty("downloadTooLarge", true);
            } else if (file->write(chunk) != chunk.size()) {
                reply->setProperty("downloadFailed", true);
            }
        }
        const bool stale = reply->property("endpoint").toString() != endpoint_->text().trimmed() ||
            reply->property("token").toString() != token_->text();
        const bool okay = !stale && !reply->property("downloadFailed").toBool() &&
            reply->error() == QNetworkReply::NoError && file && file->flush();
        QString suggested = kind == QStringLiteral("wheel")
            ? QStringLiteral("model.whl") : QStringLiteral("weights.safetensors");
        const QByteArray disposition = reply->rawHeader("Content-Disposition");
        const QRegularExpression filenamePattern(QStringLiteral("filename\\*?=(?:UTF-8''|\\\"?)([^\\\";]+)"),
                                                   QRegularExpression::CaseInsensitiveOption);
        const QRegularExpressionMatch filenameMatch = filenamePattern.match(QString::fromLatin1(disposition));
        if (filenameMatch.hasMatch()) suggested = QUrl::fromPercentEncoding(filenameMatch.captured(1).toUtf8());
        if (!okay) {
            if (!stale) {
                if (reply->property("downloadTooLarge").toBool()) {
                    setMessage(tr("Artifact exceeds the 1 GiB download limit."), true);
                } else {
                    file->seek(0);
                    setMessage(backendError(file->read(2048), reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt(), error), true);
                }
            }
            reply->deleteLater();
            return;
        }
        if (kind == QStringLiteral("wheel") && !filenameMatch.hasMatch()) {
            file->close();
            setMessage(tr("Backend wheel response did not include its valid filename."), true);
            reply->deleteLater();
            return;
        }
        suggested = QFileInfo(suggested).fileName();
        const QString destination = QFileDialog::getSaveFileName(this, tr("Download %1").arg(kind), suggested);
        if (destination.isEmpty()) { file->close(); reply->deleteLater(); return; }
        if (!file->seek(0)) { file->close(); setMessage(tr("Cannot read temporary download."), true); reply->deleteLater(); return; }
        QSaveFile output(destination);
        bool written = output.open(QIODevice::WriteOnly);
        while (written && !file->atEnd()) {
            const QByteArray chunk = file->read(1024 * 1024);
            if (chunk.isEmpty() || output.write(chunk) != chunk.size()) written = false;
        }
        file->close();
        written = written && output.commit();
        reply->deleteLater();
        if (!written) { output.cancelWriting(); setMessage(tr("Cannot save downloaded %1.").arg(kind), true); return; }
        setMessage(tr("Saved %1.").arg(QFileInfo(destination).fileName()));
    });
}

void BackendDialog::restoreSnapshot()
{
    if (selectedJob_.isEmpty()) { setMessage(tr("Select a job first."), true); return; }
    const QUrl url = endpointUrl(QStringLiteral("/v1/jobs/%1/snapshot").arg(selectedJob_));
    QNetworkRequest req(url);
    if (!token_->text().isEmpty()) req.setRawHeader("Authorization", QByteArrayLiteral("Bearer ") + token_->text().toUtf8());
    QNetworkReply *reply = network_->get(req);
    reply->setProperty("endpoint", endpoint_->text().trimmed());
    reply->setProperty("token", token_->text());
    reply->setProperty("job", selectedJob_);
    auto *timeout = new QTimer(reply); timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, reply, [reply] { reply->abort(); }); timeout->start(RequestTimeoutMs);
    connect(reply, &QNetworkReply::downloadProgress, this, [reply](qint64 received, qint64) { if (received > MaxJsonReplyBytes) reply->abort(); });
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        const QByteArray bytes = reply->readAll();
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const bool stale = reply->property("endpoint").toString() != endpoint_->text().trimmed() ||
            reply->property("token").toString() != token_->text() ||
            reply->property("job").toString() != selectedJob_;
        if (stale) { reply->deleteLater(); return; }
        if (reply->error() != QNetworkReply::NoError || status < 200 || status >= 300 || bytes.size() > MaxJsonReplyBytes) {
            setMessage(backendError(bytes, status, reply->errorString()), true); reply->deleteLater(); return;
        }
        reply->deleteLater();
        QJsonParseError error;
        const QJsonDocument doc = QJsonDocument::fromJson(bytes, &error);
        const QJsonObject snapshot = doc.object();
        if (!doc.isObject() || !snapshot.value(QStringLiteral("project")).isObject() ||
            !snapshot.value(QStringLiteral("files")).isObject()) {
            setMessage(tr("Invalid snapshot response: %1").arg(error.errorString()), true); return;
        }
        const QString parent = QFileDialog::getExistingDirectory(this, tr("Choose parent for restored project"));
        if (parent.isEmpty()) return;
        const QString name = QInputDialog::getText(this, tr("Restore snapshot"), tr("New project directory name"),
            QLineEdit::Normal, QStringLiteral("restored-job-%1").arg(selectedJob_)).trimmed();
        if (name.isEmpty() || name == QStringLiteral(".") || name == QStringLiteral("..") || name.contains('/') || name.contains('\\')) {
            setMessage(tr("Choose a simple new directory name."), true); return;
        }
        const QString destination = QDir(parent).filePath(name);
        if (QFileInfo::exists(destination)) { setMessage(tr("Restore destination already exists."), true); return; }
        QJsonParseError modelError;
        const QByteArray modelBytes = QJsonDocument(snapshot.value(QStringLiteral("project")).toObject()).toJson(QJsonDocument::Indented);
        if (modelBytes.isEmpty() || modelBytes.size() > MaxModelBytes ||
            !QJsonDocument::fromJson(modelBytes, &modelError).isObject()) {
            setMessage(tr("Snapshot model is invalid or too large."), true); return;
        }
        QDir root;
        if (!root.mkdir(destination)) { setMessage(tr("Cannot create a new restore directory."), true); return; }
        bool okay = true;
        qint64 total = 0;
        const QJsonObject files = snapshot.value(QStringLiteral("files")).toObject();
        if (files.size() > MaxResourceFiles) okay = false;
        for (auto it = files.begin(); okay && it != files.end(); ++it) {
            if (!safeRelativePath(it.key()) || it.key() == QStringLiteral("model.json") || !it.value().isString()) { okay = false; break; }
            const QString encoded = it.value().toString();
            const qint64 maxEncodedFileBytes = ((MaxResourceFileBytes + 2) / 3) * 4;
            if (encoded.size() > maxEncodedFileBytes) { okay = false; break; }
            const QByteArray decoded = QByteArray::fromBase64(encoded.toLatin1(), QByteArray::AbortOnBase64DecodingErrors);
            if (decoded.size() > MaxResourceFileBytes || decoded.size() > MaxResourceBytes - total ||
                QString::fromLatin1(decoded.toBase64()) != encoded) { okay = false; break; }
            total += decoded.size();
            const QString target = QDir(destination).filePath(it.key());
            const QString parentDirectory = QFileInfo(target).path();
            if (!QDir().mkpath(parentDirectory)) { okay = false; break; }
            QSaveFile file(target);
            if (!file.open(QIODevice::WriteOnly) || file.write(decoded) != decoded.size() || !file.commit()) { okay = false; break; }
        }
        if (okay) {
            QSaveFile modelFile(QDir(destination).filePath(QStringLiteral("model.json")));
            okay = modelFile.open(QIODevice::WriteOnly) && modelFile.write(modelBytes) == modelBytes.size() && modelFile.commit();
        }
        if (!okay) {
            QDir(destination).removeRecursively();
            setMessage(tr("Snapshot contains an unsafe path or invalid base64 data; restore was rolled back."), true);
            return;
        }
        if (!openProject_ || !openProject_(destination)) return;
        setMessage(tr("Snapshot restored and opened from %1.").arg(destination));
    });
}
