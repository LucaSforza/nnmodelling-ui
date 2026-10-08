#include "BackendDialog.hpp"

#include "application/application.h"
#include "project/project.h"
#include "BackendServiceManager.hpp"

#include <QDir>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleValidator>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QInputDialog>
#include <QIntValidator>
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
#include <QSettings>
#include <QStandardPaths>
#include <QGroupBox>
#include <QSplitter>
#include <QSignalBlocker>
#include <QSet>
#include <QTemporaryFile>
#include <QTimer>
#include <QTreeWidget>
#include <QVBoxLayout>

#include "TrainingCurveWidget.hpp"

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
    const QJsonArray steps = metrics.value(QStringLiteral("steps")).toArray();
    const bool hasSteps = !steps.isEmpty();
    lines << (hasSteps ? QObject::tr("Published step metrics:") : QObject::tr("Epoch metrics (legacy):"));
    const QJsonArray points = hasSteps ? steps : metrics.value(QStringLiteral("epochs")).toArray();
    const int first = qMax(0, points.size() - 5000);
    if (first > 0) lines << QObject::tr("Showing the latest %1 of %2 metric points.").arg(points.size() - first).arg(points.size());
    for (int index = first; index < points.size(); ++index) {
        const QJsonValue value = points.at(index);
        const QJsonObject row = value.toObject();
        const QString location = hasSteps
            ? QObject::tr("Step %1 (epoch %2)").arg(row.value(QStringLiteral("step")).toVariant().toString())
                .arg(row.value(QStringLiteral("epoch")).toVariant().toString())
            : QObject::tr("Epoch %1").arg(row.value(QStringLiteral("epoch")).toVariant().toString());
        lines << QObject::tr("%1 — training %2, validation %3")
            .arg(location)
            .arg(row.value(QStringLiteral("training_loss")).toVariant().toString(),
                 row.value(QStringLiteral("validation_loss")).toVariant().toString());
    }
    const QJsonValue testLoss = metrics.value(QStringLiteral("test_loss"));
    lines << QObject::tr("Final test loss: %1").arg(testLoss.isNull() ? QObject::tr("pending") : testLoss.toVariant().toString());
    return lines.join(QLatin1Char('\n'));
}

void updateLog(QPlainTextEdit *log, const QString &text)
{
    if (log->toPlainText() == text) return;
    QScrollBar *scroll = log->verticalScrollBar();
    const bool followBottom = scroll->value() >= scroll->maximum();
    const int position = scroll->value();
    log->setPlainText(text);
    scroll->setValue(followBottom ? scroll->maximum() : qMin(position, scroll->maximum()));
}

QString lossLabel(const QJsonValue &value)
{
    if (value.isNull() || value.isUndefined()) return QObject::tr("pending");
    if (!value.isDouble() || !std::isfinite(value.toDouble())) return QObject::tr("invalid");
    return QString::number(value.toDouble(), 'g', 8);
}

QString jobHistoryLabel(const QString &id, const QString &createdAt)
{
    const QString compactTime = createdAt.left(16).replace(QLatin1Char('T'), QLatin1Char(' '));
    const QString shortId = id.size() > 12 ? id.right(8) : id;
    return QObject::tr("%1 · %2").arg(compactTime, shortId);
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
                             std::function<bool(const QString &)> openProject, QWidget *parent,
                             BackendServiceManager *serviceManager)
    : QDialog(parent), application_(application), saveProject_(std::move(saveProject)),
      openProject_(std::move(openProject)), serviceManager_(serviceManager),
      network_(new QNetworkAccessManager(this))
{
    setWindowTitle(tr("Training backend"));
    resize(1120, 850);
    setMinimumSize(920, 680);
    QFont dialogFont = font();
    dialogFont.setPointSize(qMax(dialogFont.pointSize(), 10));
    setFont(dialogFont);
    auto *layout = new QVBoxLayout(this);
    auto *connection = new QHBoxLayout;
    connection->addWidget(new QLabel(tr("Endpoint"), this));
    endpoint_ = new QLineEdit(QStringLiteral("http://127.0.0.1:8765"), this);
    endpoint_->setObjectName(QStringLiteral("backendEndpoint"));
    endpoint_->setMinimumWidth(210);
    connection->addWidget(endpoint_, 2);
    connection->addWidget(new QLabel(tr("Bearer token"), this));
    token_ = new QLineEdit(this);
    token_->setObjectName(QStringLiteral("backendToken"));
    token_->setEchoMode(QLineEdit::Password);
    connection->addWidget(token_, 1);
    auto *connectButton = new QPushButton(tr("Connect / check health"), this);
    connectButton->setObjectName(QStringLiteral("backendConnect"));
    auto *refreshButton = new QPushButton(tr("Refresh jobs"), this);
    refreshButton->setObjectName(QStringLiteral("backendRefresh"));
    connection->addWidget(connectButton);
    connection->addWidget(refreshButton);
    layout->addLayout(connection);
    status_ = new QLabel(tr("Not connected"), this);
    status_->setObjectName(QStringLiteral("backendStatus"));
    status_->setWordWrap(true);
    layout->addWidget(status_);

    auto *launcher = new QGroupBox(tr("Local backend service configuration"), this);
    auto *launcherLayout = new QVBoxLayout(launcher);
    auto *launcherForm = new QFormLayout;
    launcherForm->setHorizontalSpacing(12);
    launcherForm->setVerticalSpacing(7);
    executor_ = new QComboBox(launcher);
    executor_->setObjectName(QStringLiteral("backendExecutor"));
    executor_->addItem(tr("Docker-compatible runtime"), QStringLiteral("docker"));
    executor_->addItem(tr("SSH / Slurm"), QStringLiteral("slurm"));
    jobRoot_ = new QLineEdit(launcher);
    jobRoot_->setObjectName(QStringLiteral("backendJobRoot"));
    dockerRuntime_ = new QLineEdit(QStringLiteral("docker"), launcher);
    dockerRuntime_->setObjectName(QStringLiteral("backendDockerRuntime"));
    dockerImage_ = new QLineEdit(QStringLiteral("nnmodelling-worker:local"), launcher);
    dockerImage_->setObjectName(QStringLiteral("backendDockerImage"));
    slurmHost_ = new QLineEdit(QStringLiteral("cluster"), launcher);
    slurmHost_->setObjectName(QStringLiteral("backendSlurmHost"));
    slurmRoot_ = new QLineEdit(launcher);
    slurmRoot_->setObjectName(QStringLiteral("backendSlurmRoot"));
    slurmImage_ = new QLineEdit(launcher);
    slurmImage_->setObjectName(QStringLiteral("backendSlurmImage"));
    slurmPartition_ = new QLineEdit(QStringLiteral("students"), launcher);
    slurmPartition_->setObjectName(QStringLiteral("backendSlurmPartition"));
    slurmCpus_ = new QLineEdit(QStringLiteral("2"), launcher);
    slurmCpus_->setObjectName(QStringLiteral("backendSlurmCpus"));
    slurmMemory_ = new QLineEdit(QStringLiteral("4G"), launcher);
    slurmMemory_->setObjectName(QStringLiteral("backendSlurmMemory"));
    slurmTime_ = new QLineEdit(QStringLiteral("00:30:00"), launcher);
    slurmTime_->setObjectName(QStringLiteral("backendSlurmTime"));
    const QString appData = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    const QString defaultRoot = appData.isEmpty()
        ? QDir::homePath() + QStringLiteral("/.local/share/nnmodelling/jobs")
        : QDir(appData).filePath(QStringLiteral("jobs"));
    jobRoot_->setText(defaultRoot);
    launcherForm->addRow(tr("Executor"), executor_);
    launcherForm->addRow(tr("Local job root"), jobRoot_);
    launcherForm->addRow(tr("Docker runtime"), dockerRuntime_);
    launcherForm->addRow(tr("Docker worker image"), dockerImage_);
    launcherForm->addRow(tr("Slurm SSH alias"), slurmHost_);
    launcherForm->addRow(tr("Remote job root"), slurmRoot_);
    launcherForm->addRow(tr("Prebuilt SIF image"), slurmImage_);
    launcherForm->addRow(tr("Partition"), slurmPartition_);
    launcherForm->addRow(tr("CPUs"), slurmCpus_);
    launcherForm->addRow(tr("Memory"), slurmMemory_);
    launcherForm->addRow(tr("Time limit"), slurmTime_);
    launcherLayout->addLayout(launcherForm);
    for (QLineEdit *field : {jobRoot_, dockerRuntime_, dockerImage_, slurmHost_, slurmRoot_,
                             slurmImage_, slurmPartition_, slurmCpus_, slurmMemory_, slurmTime_})
        field->setMinimumHeight(27);
    executor_->setMinimumHeight(27);
    auto setRowVisible = [launcherForm](QWidget *field, bool visible) {
        field->setVisible(visible);
        if (QWidget *label = launcherForm->labelForField(field)) label->setVisible(visible);
    };
    auto updateExecutorFields = [this, setRowVisible] {
        const bool docker = executor_->currentData().toString() == QStringLiteral("docker");
        setRowVisible(dockerRuntime_, docker);
        setRowVisible(dockerImage_, docker);
        setRowVisible(slurmHost_, !docker);
        setRowVisible(slurmRoot_, !docker);
        setRowVisible(slurmImage_, !docker);
        setRowVisible(slurmPartition_, !docker);
        setRowVisible(slurmCpus_, !docker);
        setRowVisible(slurmMemory_, !docker);
        setRowVisible(slurmTime_, !docker);
    };
    connect(executor_, &QComboBox::currentIndexChanged, this, updateExecutorFields);
    updateExecutorFields();
    managedBackendButton_ = new QPushButton(tr("Save settings and start local backend"), launcher);
    managedBackendButton_->setObjectName(QStringLiteral("backendManagedStart"));
    managedBackendButton_->setEnabled(serviceManager_ != nullptr);
    launcherLayout->addWidget(managedBackendButton_);
    layout->addWidget(launcher);
    QSettings launcherSettings;
    launcherSettings.beginGroup(QStringLiteral("backendLauncher"));
    executor_->setCurrentIndex(qMax(0, executor_->findData(launcherSettings.value(QStringLiteral("executor"), QStringLiteral("docker")))));
    jobRoot_->setText(launcherSettings.value(QStringLiteral("jobRoot"), jobRoot_->text()).toString());
    dockerRuntime_->setText(launcherSettings.value(QStringLiteral("dockerRuntime"), dockerRuntime_->text()).toString());
    dockerImage_->setText(launcherSettings.value(QStringLiteral("dockerImage"), dockerImage_->text()).toString());
    slurmHost_->setText(launcherSettings.value(QStringLiteral("slurmHost"), slurmHost_->text()).toString());
    slurmRoot_->setText(launcherSettings.value(QStringLiteral("slurmRoot")).toString());
    slurmImage_->setText(launcherSettings.value(QStringLiteral("slurmImage")).toString());
    slurmPartition_->setText(launcherSettings.value(QStringLiteral("slurmPartition"), slurmPartition_->text()).toString());
    slurmCpus_->setText(launcherSettings.value(QStringLiteral("slurmCpus"), slurmCpus_->text()).toString());
    slurmMemory_->setText(launcherSettings.value(QStringLiteral("slurmMemory"), slurmMemory_->text()).toString());
    slurmTime_->setText(launcherSettings.value(QStringLiteral("slurmTime"), slurmTime_->text()).toString());

    auto *training = new QHBoxLayout;
    training->addWidget(new QLabel(tr("Epochs"), this));
    epochs_ = new QLineEdit(QStringLiteral("10"), this);
    epochs_->setValidator(new QIntValidator(1, 10000, epochs_));
    epochs_->setMaximumWidth(72);
    batchSize_ = new QLineEdit(QStringLiteral("32"), this);
    batchSize_->setValidator(new QIntValidator(1, 4096, batchSize_));
    batchSize_->setMaximumWidth(72);
    learningRate_ = new QLineEdit(QStringLiteral("0.001"), this);
    seed_ = new QLineEdit(QStringLiteral("0"), this);
    seed_->setMaximumWidth(92);
    publishEverySteps_ = new QLineEdit(QStringLiteral("10"), this);
    publishEverySteps_->setObjectName(QStringLiteral("trainingPublishEverySteps"));
    publishEverySteps_->setValidator(new QIntValidator(1, 100000, publishEverySteps_));
    publishEverySteps_->setMaximumWidth(86);
    epochs_->setObjectName(QStringLiteral("trainingEpochs"));
    batchSize_->setObjectName(QStringLiteral("trainingBatchSize"));
    learningRate_->setObjectName(QStringLiteral("trainingLearningRate"));
    seed_->setObjectName(QStringLiteral("trainingSeed"));
    training->addWidget(epochs_);
    training->addWidget(new QLabel(tr("Batch"), this));
    training->addWidget(batchSize_);
    training->addWidget(new QLabel(tr("Learning rate"), this));
    learningRate_->setMaximumWidth(110);
    learningRate_->setValidator(new QDoubleValidator(0.0, 1.0, 12, learningRate_));
    training->addWidget(learningRate_);
    training->addWidget(new QLabel(tr("Seed"), this));
    training->addWidget(seed_);
    training->addWidget(new QLabel(tr("Publish every N steps"), this));
    training->addWidget(publishEverySteps_);
    auto *submitButton = new QPushButton(tr("Save project and submit"), this);
    submitButton->setObjectName(QStringLiteral("backendSubmit"));
    training->addWidget(submitButton);
    training->addStretch(1);
    layout->addLayout(training);

    auto *workspace = new QSplitter(Qt::Horizontal, this);
    auto *historyPanel = new QWidget(workspace);
    historyPanel->setMinimumWidth(250);
    auto *historyLayout = new QVBoxLayout(historyPanel);
    historyLayout->setContentsMargins(0, 0, 6, 0);
    auto *historyTitle = new QLabel(tr("Job history"), historyPanel);
    historyTitle->setStyleSheet(QStringLiteral("font-weight: 600;"));
    historyLayout->addWidget(historyTitle);
    jobs_ = new QTreeWidget(this);
    jobs_->setObjectName(QStringLiteral("backendJobs"));
    jobs_->setColumnCount(2);
    jobs_->setHeaderLabels({tr("Job · created"), tr("Status")});
    jobs_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    jobs_->header()->setStretchLastSection(false);
    jobs_->header()->setSectionResizeMode(0, QHeaderView::Stretch);
    jobs_->header()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    historyLayout->addWidget(jobs_, 1);
    auto *rightPanel = new QWidget(workspace);
    auto *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(6, 0, 0, 0);
    auto *curveControls = new QHBoxLayout;
    auto *curveTitle = new QLabel(tr("Learning curves"), rightPanel);
    curveTitle->setStyleSheet(QStringLiteral("font-weight: 600;"));
    curveControls->addWidget(curveTitle);
    curveControls->addSpacing(10);
    trainingCurveVisible_ = new QCheckBox(tr("Training loss"), rightPanel);
    trainingCurveVisible_->setObjectName(QStringLiteral("curveTrainingVisible"));
    trainingCurveVisible_->setChecked(true);
    validationCurveVisible_ = new QCheckBox(tr("Validation loss"), rightPanel);
    validationCurveVisible_->setObjectName(QStringLiteral("curveValidationVisible"));
    validationCurveVisible_->setChecked(true);
    trainingCurveVisible_->setStyleSheet(QStringLiteral("color: #2677b8; font-weight: 600;"));
    validationCurveVisible_->setStyleSheet(QStringLiteral("color: #d75252; font-weight: 600;"));
    curveControls->addWidget(trainingCurveVisible_);
    curveControls->addWidget(validationCurveVisible_);
    curveControls->addSpacing(12);
    curveControls->addWidget(new QLabel(tr("Scale"), rightPanel));
    curveScale_ = new QComboBox(rightPanel);
    curveScale_->setObjectName(QStringLiteral("curveScale"));
    curveScale_->addItem(tr("Linear"), false);
    curveScale_->addItem(tr("Log"), true);
    curveControls->addWidget(curveScale_);
    curveControls->addStretch(1);
    finalTestLoss_ = new QLabel(tr("Final test loss: pending"), rightPanel);
    finalTestLoss_->setObjectName(QStringLiteral("finalTestLoss"));
    curveControls->addWidget(finalTestLoss_);
    rightLayout->addLayout(curveControls);

    curve_ = new TrainingCurveWidget(rightPanel);
    rightLayout->addWidget(curve_, 3);
    auto *logTitle = new QLabel(tr("Training and worker log"), rightPanel);
    logTitle->setStyleSheet(QStringLiteral("font-weight: 600;"));
    rightLayout->addWidget(logTitle);
    metrics_ = new QPlainTextEdit(rightPanel);
    metrics_->setObjectName(QStringLiteral("backendMetrics"));
    metrics_->setReadOnly(true);
    metrics_->setMinimumHeight(115);
    rightLayout->addWidget(metrics_, 1);
    workspace->addWidget(historyPanel);
    workspace->addWidget(rightPanel);
    workspace->setStretchFactor(0, 0);
    workspace->setStretchFactor(1, 1);
    workspace->setSizes({290, 780});
    layout->addWidget(workspace, 1);

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
    closeButton->setObjectName(QStringLiteral("backendClose"));
    actions->addWidget(closeButton);
    layout->addLayout(actions);
    connect(closeButton, &QPushButton::clicked, this, &QDialog::accept);
    connect(connectButton, &QPushButton::clicked, this, &BackendDialog::connectBackend);
    connect(refreshButton, &QPushButton::clicked, this, &BackendDialog::refreshJobs);
    connect(submitButton, &QPushButton::clicked, this, &BackendDialog::submitJob);
    connect(managedBackendButton_, &QPushButton::clicked, this, &BackendDialog::startManagedBackend);
    if (serviceManager_) {
        connect(serviceManager_, &BackendServiceManager::processError, this, [this](const QString &message) {
            setMessage(message, true);
        });
        if (serviceManager_->isRunning()) {
            endpoint_->setText(serviceManager_->endpoint().toString());
            token_->setText(serviceManager_->bearerToken());
            managedBackendButton_->setText(tr("Restart local backend safely"));
            QTimer::singleShot(500, this, &BackendDialog::connectBackend);
        }
    }
    connect(trainingCurveVisible_, &QCheckBox::toggled, curve_, &TrainingCurveWidget::setTrainingVisible);
    connect(validationCurveVisible_, &QCheckBox::toggled, curve_, &TrainingCurveWidget::setValidationVisible);
    connect(curveScale_, &QComboBox::currentIndexChanged, this, [this](int index) {
        curve_->setLogarithmic(curveScale_->itemData(index).toBool());
    });
    connect(jobs_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem *current) {
        if (!current) return;
        selectedJob_ = current->data(0, Qt::UserRole).toString();
        curve_->setMetrics({});
        finalTestLoss_->setText(tr("Final test loss: pending"));
        updateLog(metrics_, tr("Loading selected job…"));
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
    base.setQuery(QString());
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
        const QJsonObject object = document.object();
        if (path == QStringLiteral("/health")) {
            const QJsonObject container = object.value(QStringLiteral("container")).toObject();
            const QString runtime = container.value(QStringLiteral("runtime")).toString(tr("container runtime"));
            if (container.value(QStringLiteral("available")).toBool()) {
                setMessage(tr("Connected — %1 is ready for training.").arg(runtime));
            } else {
                const QString detail = container.value(QStringLiteral("error")).toString(tr("runtime is unavailable"));
                setMessage(tr("Backend connected, but training is unavailable (%1): %2").arg(runtime, detail), true);
            }
            refreshJobs();
        } else if (path == QStringLiteral("/v1/jobs")) {
            const QString oldSelection = selectedJob_;
            const QSignalBlocker blocker(jobs_);
            jobs_->clear();
            for (const QJsonValue &entry : object.value(QStringLiteral("jobs")).toArray()) {
                const QJsonObject job = entry.toObject();
                const QString id = job.value(QStringLiteral("id")).toString();
                const QString createdAt = job.value(QStringLiteral("created_at")).toString();
                auto *item = new QTreeWidgetItem(jobs_, {jobHistoryLabel(id, createdAt),
                                                          job.value(QStringLiteral("status")).toString()});
                item->setData(0, Qt::UserRole, id);
                item->setToolTip(0, tr("Job ID: %1\nCreated: %2").arg(id, createdAt));
                if (id == oldSelection) jobs_->setCurrentItem(item);
            }
            if (!selectedJob_.isEmpty()) showJob(selectedJob_);
        } else if (path.contains(QStringLiteral("/cancel"))) {
            setMessage(tr("Cancellation request completed."));
            showJob(selectedJob_);
            refreshJobs();
        } else if (path.endsWith(QStringLiteral("/snapshot"))) {
            // Handled by restoreSnapshot through a one-shot callback below.
        } else if (method == QByteArrayLiteral("POST") && object.contains(QStringLiteral("id"))) {
            selectedJob_ = object.value(QStringLiteral("id")).toString();
            setMessage(tr("Job submitted: %1").arg(selectedJob_));
            refreshJobs();
            showJob(selectedJob_);
        } else if (path == QStringLiteral("/v1/jobs/%1").arg(selectedJob_)) {
            const QJsonObject metrics = object.value(QStringLiteral("metrics")).toObject();
            curve_->setMetrics(metrics);
            finalTestLoss_->setText(tr("Final test loss: %1").arg(lossLabel(metrics.value(QStringLiteral("test_loss")))));
            updateLog(metrics_, formatJob(object));
        }
    });
}

void BackendDialog::connectBackend()
{
    request(QStringLiteral("/health"));
}

void BackendDialog::saveLauncherSettings()
{
    QSettings settings;
    settings.beginGroup(QStringLiteral("backendLauncher"));
    settings.setValue(QStringLiteral("executor"), executor_->currentData().toString());
    settings.setValue(QStringLiteral("jobRoot"), QDir::cleanPath(jobRoot_->text().trimmed()));
    settings.setValue(QStringLiteral("dockerRuntime"), dockerRuntime_->text().trimmed());
    settings.setValue(QStringLiteral("dockerImage"), dockerImage_->text().trimmed());
    settings.setValue(QStringLiteral("slurmHost"), slurmHost_->text().trimmed());
    settings.setValue(QStringLiteral("slurmRoot"), slurmRoot_->text().trimmed());
    settings.setValue(QStringLiteral("slurmImage"), slurmImage_->text().trimmed());
    settings.setValue(QStringLiteral("slurmPartition"), slurmPartition_->text().trimmed());
    settings.setValue(QStringLiteral("slurmCpus"), slurmCpus_->text().trimmed());
    settings.setValue(QStringLiteral("slurmMemory"), slurmMemory_->text().trimmed());
    settings.setValue(QStringLiteral("slurmTime"), slurmTime_->text().trimmed());
}

void BackendDialog::startManagedBackend()
{
    if (!serviceManager_) return;
    const QString root = QDir::cleanPath(jobRoot_->text().trimmed());
    if (!QDir::isAbsolutePath(root) || root.contains(QChar::Null) || root.contains(QChar::LineFeed)) {
        setMessage(tr("Local job root must be an absolute path."), true);
        return;
    }
    QStringList environment;
    environment << QStringLiteral("NNMODELLING_EXECUTOR=") + executor_->currentData().toString()
                << QStringLiteral("NNMODELLING_JOB_ROOT=") + root;
    if (!token_->text().isEmpty())
        environment << QStringLiteral("NNMODELLING_BEARER_TOKEN=") + token_->text();
    if (executor_->currentData().toString() == QStringLiteral("docker")) {
        const QString runtime = dockerRuntime_->text().trimmed();
        const QString image = dockerImage_->text().trimmed();
        static const QRegularExpression runtimePattern(QStringLiteral("^[A-Za-z0-9_./+-]+$"));
        static const QRegularExpression imagePattern(QStringLiteral("^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,254}$"));
        if (!runtimePattern.match(runtime).hasMatch() || !imagePattern.match(image).hasMatch()) {
            setMessage(tr("Docker runtime and worker image must be valid executable and image names."), true);
            return;
        }
        environment << QStringLiteral("NNMODELLING_CONTAINER_RUNTIME=") + runtime
                    << QStringLiteral("NNMODELLING_WORKER_IMAGE=") + image;
    } else {
        const QString host = slurmHost_->text().trimmed();
        const QString remoteRoot = slurmRoot_->text().trimmed();
        const QString image = slurmImage_->text().trimmed();
        const QString partition = slurmPartition_->text().trimmed();
        bool cpusOk = false;
        const int cpus = slurmCpus_->text().trimmed().toInt(&cpusOk);
        static const QRegularExpression aliasPattern(QStringLiteral("^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"));
        static const QRegularExpression remotePathPattern(QStringLiteral("^/(?:[^\\n\\r]+)$"));
        static const QRegularExpression tokenPattern(QStringLiteral("^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"));
        static const QRegularExpression memoryPattern(QStringLiteral("^[1-9][0-9]*[KMGTP]?$"));
        static const QRegularExpression timePattern(QStringLiteral("^[0-9]{1,3}(?:-[0-9]{1,2})?:[0-9]{2}:[0-9]{2}$"));
        const QString memory = slurmMemory_->text().trimmed();
        const QString time = slurmTime_->text().trimmed();
        if (!aliasPattern.match(host).hasMatch() || !QDir::isAbsolutePath(remoteRoot) ||
            remoteRoot.split(QLatin1Char('/')).contains(QStringLiteral("..")) ||
            !remotePathPattern.match(image).hasMatch() || image.split(QLatin1Char('/')).contains(QStringLiteral("..")) ||
            !tokenPattern.match(partition).hasMatch() || !cpusOk || cpus < 1 || cpus > 256 ||
            !memoryPattern.match(memory).hasMatch() || !timePattern.match(time).hasMatch()) {
            setMessage(tr("Slurm needs an SSH alias, absolute remote root and SIF path, valid partition, CPUs, memory and time limit."), true);
            return;
        }
        environment << QStringLiteral("NNMODELLING_SLURM_HOST=") + host
                    << QStringLiteral("NNMODELLING_SLURM_ROOT=") + remoteRoot
                    << QStringLiteral("NNMODELLING_SLURM_IMAGE=") + image
                    << QStringLiteral("NNMODELLING_SLURM_PARTITION=") + partition
                    << QStringLiteral("NNMODELLING_SLURM_CPUS=") + QString::number(cpus)
                    << QStringLiteral("NNMODELLING_SLURM_MEMORY=") + memory
                    << QStringLiteral("NNMODELLING_SLURM_TIME=") + time;
    }
    saveLauncherSettings();
    if (serviceManager_->isRunning()) {
        checkJobsBeforeRestart();
        return;
    }
    QString error;
    if (!serviceManager_->start(environment, &error)) {
        setMessage(error, true);
        return;
    }
    endpoint_->setText(serviceManager_->endpoint().toString());
    managedBackendButton_->setText(tr("Restart local backend safely"));
    setMessage(tr("Starting local backend on loopback…"));
    QTimer::singleShot(750, this, &BackendDialog::connectBackend);
}

void BackendDialog::checkJobsBeforeRestart()
{
    const QUrl url = serviceManager_->endpoint().resolved(QUrl(QStringLiteral("/v1/jobs")));
    QNetworkRequest request(url);
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    const QString token = token_->text();
    if (!token.isEmpty()) request.setRawHeader("Authorization", QByteArrayLiteral("Bearer ") + token.toUtf8());
    QNetworkReply *reply = network_->get(request);
    auto *timeout = new QTimer(reply);
    timeout->setSingleShot(true);
    connect(timeout, &QTimer::timeout, reply, [reply] { reply->abort(); });
    timeout->start(RequestTimeoutMs);
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        const QByteArray bytes = reply->readAll();
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const auto networkError = reply->error();
        const QString networkMessage = reply->errorString();
        reply->deleteLater();
        const QJsonDocument document = QJsonDocument::fromJson(bytes);
        if (networkError != QNetworkReply::NoError || status < 200 || status >= 300 || !document.isObject()) {
            setMessage(backendError(bytes, status, networkMessage), true);
            return;
        }
        for (const QJsonValue &entry : document.object().value(QStringLiteral("jobs")).toArray()) {
            const QString state = entry.toObject().value(QStringLiteral("status")).toString();
            if (state == QStringLiteral("queued") || state == QStringLiteral("running")) {
                setMessage(tr("The managed backend cannot restart while jobs are queued or running."), true);
                return;
            }
        }
        serviceManager_->stop();
        startManagedBackend();
    });
}

void BackendDialog::refreshJobs()
{
    request(QStringLiteral("/v1/jobs"));
}

void BackendDialog::submitJob()
{
    bool epochsOk = false, batchOk = false, seedOk = false, lrOk = false, publishOk = false;
    const int epochs = epochs_->text().toInt(&epochsOk);
    const int batch = batchSize_->text().toInt(&batchOk);
    const qint64 seed = seed_->text().toLongLong(&seedOk);
    const double learningRate = learningRate_->text().toDouble(&lrOk);
    const int publishEverySteps = publishEverySteps_->text().toInt(&publishOk);
    if (!epochsOk || epochs < 1 || epochs > 10000 || !batchOk || batch < 1 || batch > 4096 ||
        !seedOk || seed < -2147483648LL || seed > 4294967295LL || !lrOk ||
        !std::isfinite(learningRate) || learningRate <= 0.0 || learningRate > 1.0 ||
        !publishOk || publishEverySteps < 1 || publishEverySteps > 100000) {
        setMessage(tr("Training settings require valid epochs, batch size, learning rate, seed and publication cadence."), true);
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
            {QStringLiteral("seed"), static_cast<double>(seed)},
            {QStringLiteral("publish_every_steps"), publishEverySteps}}}
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
