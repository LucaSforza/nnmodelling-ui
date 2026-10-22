#include "BackendDialog.hpp"
#include "BackendServiceManager.hpp"
#include "TrainingCurveWidget.hpp"

#include "application/application.h"

#include <QDir>
#include <QComboBox>
#include <QImage>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QFileInfo>
#include <QTest>
#include <QSignalSpy>
#include <QTreeWidget>

class FakeBackend final : public QObject {
    Q_OBJECT
public:
    QTcpServer server;
    int requestCount = 0;
    int submitCount = 0;
    int detailCount = 0;
    QJsonObject submitted;
    QByteArray authorization;

    FakeBackend()
    {
        connect(&server, &QTcpServer::newConnection, this, [this] {
            while (server.hasPendingConnections()) {
                QTcpSocket *socket = server.nextPendingConnection();
                connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
                    QByteArray buffer = socket->property("buffer").toByteArray();
                    buffer += socket->readAll();
                    socket->setProperty("buffer", buffer);
                    const qsizetype boundary = buffer.indexOf("\r\n\r\n");
                    if (boundary < 0) return;
                    const QByteArray headers = buffer.left(boundary);
                    const QByteArray firstLine = headers.left(headers.indexOf("\r\n"));
                    const QByteArray method = firstLine.left(firstLine.indexOf(' '));
                    const QByteArray path = firstLine.mid(firstLine.indexOf(' ') + 1,
                        firstLine.lastIndexOf(' ') - firstLine.indexOf(' ') - 1);
                    qsizetype contentLength = 0;
                    for (const QByteArray &line : headers.split('\n')) {
                        if (line.toLower().startsWith("authorization:")) authorization = line.mid(line.indexOf(':') + 1).trimmed();
                        if (line.toLower().startsWith("content-length:"))
                            contentLength = line.mid(line.indexOf(':') + 1).trimmed().toLongLong();
                    }
                    const QByteArray body = buffer.mid(boundary + 4);
                    if (body.size() < contentLength) return;
                    ++requestCount;
                    QByteArray responseBody;
                    int status = 200;
                    const QString existingJobId = QStringLiteral("550e8400-e29b-41d4-a716-446655440000");
                    if (path.endsWith("/health")) {
                        if (path.startsWith("/bad/")) {
                            status = 401;
                            responseBody = QByteArrayLiteral("{\"detail\":\"token rejected\"}");
                        } else if (path.startsWith("/unavailable/")) {
                            responseBody = QJsonDocument(QJsonObject{{QStringLiteral("status"), QStringLiteral("ok")},
                                {QStringLiteral("container"), QJsonObject{{QStringLiteral("available"), false},
                                    {QStringLiteral("runtime"), QStringLiteral("podman")},
                                    {QStringLiteral("error"), QStringLiteral("Container daemon is unavailable.")}}}})
                                .toJson(QJsonDocument::Compact);
                        } else {
                            responseBody = QJsonDocument(QJsonObject{{QStringLiteral("status"), QStringLiteral("ok")},
                                {QStringLiteral("container"), QJsonObject{{QStringLiteral("available"), true},
                                    {QStringLiteral("runtime"), QStringLiteral("podman")},
                                    {QStringLiteral("error"), QJsonValue(QJsonValue::Null)}}}})
                                .toJson(QJsonDocument::Compact);
                        }
                    } else if (method == "POST" && path.endsWith("/v1/jobs")) {
                        const QJsonDocument parsed = QJsonDocument::fromJson(body.left(contentLength));
                        submitted = parsed.object();
                        ++submitCount;
                        responseBody = job("submitted-1", "queued", QJsonObject{{QStringLiteral("epochs"), QJsonArray{}},
                            {QStringLiteral("test_loss"), QJsonValue(QJsonValue::Null)}});
                    } else if (path.endsWith("/v1/jobs")) {
                        QJsonArray jobs;
                        jobs.append(QJsonObject{{QStringLiteral("id"), existingJobId},
                            {QStringLiteral("status"), QStringLiteral("completed")},
                            {QStringLiteral("created_at"), QStringLiteral("2026-10-05T12:00:00Z")}});
                        jobs.append(QJsonObject{{QStringLiteral("id"), QStringLiteral("legacy-1")},
                            {QStringLiteral("status"), QStringLiteral("completed")},
                            {QStringLiteral("created_at"), QStringLiteral("2026-10-04T12:00:00Z")}});
                        responseBody = QJsonDocument(QJsonObject{{QStringLiteral("jobs"), jobs}}).toJson(QJsonDocument::Compact);
                    } else if (path.contains("/v1/jobs/")) {
                        ++detailCount;
                        if (path.endsWith("/legacy-1")) {
                            QJsonArray legacyEpochs{QJsonObject{{QStringLiteral("epoch"), 3},
                                {QStringLiteral("training_loss"), 0.3}, {QStringLiteral("validation_loss"), 0.2}}};
                            responseBody = job("legacy-1", "completed", QJsonObject{
                                {QStringLiteral("steps"), QJsonArray{}}, {QStringLiteral("epochs"), legacyEpochs},
                                {QStringLiteral("test_loss"), 0.25}});
                        } else {
                        QJsonArray epochs;
                        for (int epoch = 1; epoch <= 120; ++epoch)
                            epochs.append(QJsonObject{{QStringLiteral("epoch"), epoch},
                                {QStringLiteral("training_loss"), 0.5}, {QStringLiteral("validation_loss"), 0.4}});
                        QJsonArray steps;
                        for (int step = 1; step <= 120; ++step)
                            steps.append(QJsonObject{{QStringLiteral("step"), step}, {QStringLiteral("epoch"), 1},
                                {QStringLiteral("training_loss"), step == 1 ? 0.6 : 0.5},
                                {QStringLiteral("validation_loss"), step == 1 ? 0.5 : 0.4}});
                        responseBody = job(existingJobId, "completed", QJsonObject{{QStringLiteral("epochs"), epochs},
                            {QStringLiteral("steps"), steps},
                            {QStringLiteral("test_loss"), 0.25},
                            {QStringLiteral("early_stopping"), QJsonObject{
                                {QStringLiteral("best_epoch"), 4},
                                {QStringLiteral("best_validation_loss"), 0.2},
                                {QStringLiteral("non_improving_epochs"), 2},
                                {QStringLiteral("stopped_epoch"), 6}}}});
                        }
                    } else responseBody = QByteArrayLiteral("{}");
                    const QByteArray statusText = status == 200 ? QByteArrayLiteral("200 OK") : QByteArrayLiteral("401 Unauthorized");
                    socket->write("HTTP/1.1 " + statusText + "\r\nContent-Type: application/json\r\nContent-Length: " +
                        QByteArray::number(responseBody.size()) + "\r\nConnection: close\r\n\r\n" + responseBody);
                    socket->disconnectFromHost();
                });
            }
        });
    }

    static QByteArray job(const QString &id, const QString &status, const QJsonObject &metrics)
    {
        return QJsonDocument(QJsonObject{{QStringLiteral("id"), id},
            {QStringLiteral("status"), status},
            {QStringLiteral("created_at"), QStringLiteral("2026-10-05T12:00:00Z")},
            {QStringLiteral("error"), QJsonValue(QJsonValue::Null)},
            {QStringLiteral("metrics"), metrics}}).toJson(QJsonDocument::Compact);
    }
};

class BackendDialogTest final : public QObject {
    Q_OBJECT
private slots:
    void workerPreparationStreamsEventsAndPublishesOnlySuccess()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QByteArray oldConfig = qgetenv("NNMODELLING_CONFIG");
        const QByteArray oldPath = qgetenv("PATH");
        const bool hadConfig = qEnvironmentVariableIsSet("NNMODELLING_CONFIG");
        const bool hadPath = qEnvironmentVariableIsSet("PATH");
        struct RestoreEnvironment {
            QByteArray config, path;
            bool hadConfig, hadPath;
            ~RestoreEnvironment() {
                if (hadConfig) qputenv("NNMODELLING_CONFIG", config); else qunsetenv("NNMODELLING_CONFIG");
                if (hadPath) qputenv("PATH", path); else qunsetenv("PATH");
            }
        } restore{oldConfig, oldPath, hadConfig, hadPath};

        const QString bin = QDir(temporary.path()).filePath(QStringLiteral("bin"));
        QVERIFY(QDir().mkpath(bin));
        const QString executable = QDir(bin).filePath(QStringLiteral("uv"));
        const QString source = QDir(temporary.path()).filePath(QStringLiteral("source"));
        QVERIFY(QDir().mkpath(QDir(source).filePath(QStringLiteral("backend"))));
        QFile helper(QDir(source).filePath(QStringLiteral("backend/prepare_worker.py")));
        QVERIFY(helper.open(QIODevice::WriteOnly));
        helper.close();
        QFile runner(executable);
        QVERIFY(runner.open(QIODevice::WriteOnly));
        const QByteArray success = QByteArrayLiteral("#!/bin/sh\nprintf '%s\\n' '{\"phase\":\"build\",\"message\":\"Building image\",\"current\":1,\"total\":2}' '{\"phase\":\"success\",\"message\":\"Installed\",\"image\":\"/cluster/images/worker-hash.sif\"}'\n");
        QCOMPARE(runner.write(success), qint64(success.size()));
        runner.close();
        QVERIFY(runner.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner));

        const QString profilePath = QDir(temporary.path()).filePath(QStringLiteral("config.json"));
        QFile profileFile(profilePath);
        QVERIFY(profileFile.open(QIODevice::WriteOnly));
        const QJsonObject profile{{QStringLiteral("backend"), QJsonObject{{QStringLiteral("executor"), QStringLiteral("slurm")}}},
            {QStringLiteral("slurm"), QJsonObject{{QStringLiteral("image"), QStringLiteral("/cluster/images/old.sif")}}}};
        const QByteArray profileBytes = QJsonDocument(profile).toJson(QJsonDocument::Compact);
        QCOMPARE(profileFile.write(profileBytes), qint64(profileBytes.size()));
        profileFile.close();
        qputenv("NNMODELLING_CONFIG", profilePath.toUtf8());
        qputenv("PATH", (bin + QLatin1Char(':') + QString::fromLocal8Bit(oldPath)).toLocal8Bit());

        BackendServiceManager manager(source);
        QSignalSpy progress(&manager, &BackendServiceManager::workerPreparationProgress);
        QSignalSpy finished(&manager, &BackendServiceManager::workerPreparationFinished);
        QString error;
        QVERIFY2(manager.prepareWorker(false, {}, &error), qPrintable(error));
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 5000);
        QVERIFY(finished.takeFirst().at(0).toBool());
        QVERIFY(progress.count() >= 2);
        QJsonObject loaded;
        QString loadedPath;
        QVERIFY2(manager.loadUserConfig(&loadedPath, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.value(QStringLiteral("slurm")).toObject().value(QStringLiteral("image")).toString(),
                 QStringLiteral("/cluster/images/worker-hash.sif"));

        runner.resize(0);
        QVERIFY(runner.open(QIODevice::WriteOnly | QIODevice::Truncate));
        const QByteArray failure = QByteArrayLiteral("#!/bin/sh\nprintf '%s\\n' '{\"phase\":\"error\",\"message\":\"Check failed\",\"error\":\"Worker check failed\"}'\nexit 1\n");
        QCOMPARE(runner.write(failure), qint64(failure.size()));
        runner.close();
        QVERIFY(manager.prepareWorker(true, {}, &error));
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 5000);
        QVERIFY(!finished.takeFirst().at(0).toBool());
        QVERIFY2(manager.loadUserConfig(&loadedPath, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.value(QStringLiteral("slurm")).toObject().value(QStringLiteral("image")).toString(),
                 QStringLiteral("/cluster/images/worker-hash.sif"));

        runner.resize(0);
        QVERIFY(runner.open(QIODevice::WriteOnly | QIODevice::Truncate));
        const QByteArray cleanup = QByteArrayLiteral(
            "#!/bin/sh\n"
            "trap 'sleep 2; echo \"{\\\"phase\\\":\\\"success\\\",\\\"message\\\":\\\"late success\\\",\\\"image\\\":\\\"/cluster/images/late.sif\\\"}\"; exit 0' TERM\n"
            "echo '{\"phase\":\"build\",\"message\":\"Long cleanup\",\"current\":1,\"total\":1}'\n"
            "while :; do sleep 0.1; done\n");
        QCOMPARE(runner.write(cleanup), qint64(cleanup.size()));
        runner.close();
        QVERIFY(runner.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner));
        const int previousEvents = progress.count();
        QVERIFY(manager.prepareWorker(false, {}, &error));
        QTRY_VERIFY_WITH_TIMEOUT(manager.workerPreparationRunning(), 3000);
        QTRY_VERIFY_WITH_TIMEOUT(progress.count() > previousEvents, 3000);
        manager.cancelWorkerPreparation();
        QTRY_COMPARE_WITH_TIMEOUT(finished.count(), 1, 5000);
        QVERIFY(!finished.takeFirst().at(0).toBool());
        bool announcedCancellation = false;
        for (const QList<QVariant> &event : progress)
            if (!event.isEmpty() && event[0].toString() == QStringLiteral("cancelling"))
                announcedCancellation = true;
        QVERIFY(announcedCancellation);
        QVERIFY2(manager.loadUserConfig(&loadedPath, &loaded, &error), qPrintable(error));
        QCOMPARE(loaded.value(QStringLiteral("slurm")).toObject().value(QStringLiteral("image")).toString(),
                 QStringLiteral("/cluster/images/worker-hash.sif"));
    }

    void localBackendConfigurationValidationAndPersistence()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QByteArray priorConfig = qgetenv("NNMODELLING_CONFIG");
        const bool hadConfig = qEnvironmentVariableIsSet("NNMODELLING_CONFIG");
        struct RestoreConfigEnvironment {
            QByteArray value;
            bool wasSet;
            ~RestoreConfigEnvironment() { if (wasSet) qputenv("NNMODELLING_CONFIG", value); else qunsetenv("NNMODELLING_CONFIG"); }
        } restore{priorConfig, hadConfig};
        const QString profilePath = QDir(temporary.path()).filePath(QStringLiteral("config.json"));
        QFile initialProfile(profilePath);
        QVERIFY(initialProfile.open(QIODevice::WriteOnly));
        const QByteArray initialProfileBytes = QJsonDocument(QJsonObject{
            {QStringLiteral("backend"), QJsonObject{{QStringLiteral("executor"), QStringLiteral("docker")}}},
            {QStringLiteral("unrelated"), QJsonObject{{QStringLiteral("nested"), QJsonArray{1, 2, 3}}}}
        }).toJson(QJsonDocument::Indented);
        QCOMPARE(initialProfile.write(initialProfileBytes), qint64(initialProfileBytes.size()));
        initialProfile.close();
        qputenv("NNMODELLING_CONFIG", profilePath.toUtf8());
        BackendServiceManager manager(NN_SOURCE_DIR);
        BackendDialog dialog(nullptr, {}, {}, nullptr, &manager);
        const QStringList launcherControls{
            QStringLiteral("backendExecutor"), QStringLiteral("backendJobRoot"),
            QStringLiteral("backendDockerRuntime"), QStringLiteral("backendDockerImage"),
            QStringLiteral("backendSlurmHost"), QStringLiteral("backendSlurmRoot"),
            QStringLiteral("backendSlurmImage"), QStringLiteral("backendSlurmPartition"),
            QStringLiteral("backendSlurmCpus"), QStringLiteral("backendSlurmMemory"),
            QStringLiteral("backendSlurmTime"), QStringLiteral("backendSlurmGpus"),
            QStringLiteral("backendSlurmSshExecutable"),
            QStringLiteral("backendManagedStart")};
        for (const QString &name : launcherControls)
            QVERIFY2(!dialog.findChild<QWidget *>(name), qPrintable(name));
        dialog.show();
        dialog.findChild<QLineEdit *>(QStringLiteral("backendToken"))->setText(QStringLiteral("session-secret"));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConfigure")), Qt::LeftButton);
        auto *configuration = dialog.findChild<QDialog *>(QStringLiteral("backendConfigurationWindow"));
        QVERIFY(configuration);
        QVERIFY(configuration != &dialog);
        QVERIFY(configuration->isWindow());
        QVERIFY(!configuration->isModal());
        QVERIFY(dialog.isEnabled());
        QVERIFY(configuration->isVisible());
        auto *executor = configuration->findChild<QComboBox *>(QStringLiteral("backendExecutor"));
        auto *jobRoot = configuration->findChild<QLineEdit *>(QStringLiteral("backendJobRoot"));
        auto *startButton = configuration->findChild<QPushButton *>(QStringLiteral("backendManagedStart"));
        QVERIFY(executor);
        QVERIFY(jobRoot);
        QVERIFY(startButton);
        QVERIFY2(startButton->isEnabled(), qPrintable(configuration->findChild<QLabel *>(QStringLiteral("backendConfigurationStatus"))->text()));
        QVERIFY(configuration->findChild<QLabel *>(QStringLiteral("backendConfigurationPath")));
        QVERIFY(configuration->findChild<QLabel *>(QStringLiteral("backendConfigurationPath"))->text().contains(profilePath));
        QCOMPARE(executor->parentWidget(), configuration);
        QCOMPARE(jobRoot->parentWidget(), configuration);
        QCOMPARE(configuration->findChild<QLineEdit *>(QStringLiteral("backendSlurmHost"))->text(), QString());
        QCOMPARE(configuration->findChild<QLineEdit *>(QStringLiteral("backendSlurmPartition"))->text(), QString());
        executor->setCurrentIndex(1);
        jobRoot->setText(QStringLiteral("relative/jobs"));
        QTest::mouseClick(startButton, Qt::LeftButton);
        QVERIFY(configuration->findChild<QLabel *>(QStringLiteral("backendConfigurationStatus"))->text().contains(
            QStringLiteral("absolute path")));

        QVERIFY(!manager.isRunning());
        QJsonObject profile{
            {QStringLiteral("backend"), QJsonObject{
                {QStringLiteral("executor"), QStringLiteral("slurm")},
                {QStringLiteral("job_root"), temporary.path()},
                {QStringLiteral("container_runtime"), QStringLiteral("docker")},
                {QStringLiteral("worker_image"), QStringLiteral("nnmodelling-worker:local")}}},
            {QStringLiteral("slurm"), QJsonObject{
                {QStringLiteral("host"), QStringLiteral("test-cluster")},
                {QStringLiteral("root"), QStringLiteral("/cluster/jobs")},
                {QStringLiteral("image"), QStringLiteral("/cluster/images/worker.sif")},
                {QStringLiteral("partition"), QStringLiteral("test")},
                {QStringLiteral("cpus"), 2}, {QStringLiteral("memory"), QStringLiteral("4G")},
                {QStringLiteral("time"), QStringLiteral("00:30:00")},
                {QStringLiteral("gpus"), 3}, {QStringLiteral("ssh_executable"), QStringLiteral("ssh")}}}
        };
        QString savedPath;
        QString saveError;
        QVERIFY2(manager.saveUserConfig(profile, &savedPath, &saveError), qPrintable(saveError));
        QCOMPARE(savedPath, profilePath);
        QVERIFY(QFileInfo::exists(profilePath));
        QFile savedProfile(profilePath);
        QVERIFY(savedProfile.open(QIODevice::ReadOnly));
        const QJsonObject saved = QJsonDocument::fromJson(savedProfile.readAll()).object();
        QCOMPARE(saved.value(QStringLiteral("unrelated")).toObject().value(QStringLiteral("nested")).toArray(), QJsonArray({1, 2, 3}));
        QCOMPARE(saved.value(QStringLiteral("backend")).toObject().value(QStringLiteral("executor")).toString(), QStringLiteral("slurm"));
        QCOMPARE(saved.value(QStringLiteral("slurm")).toObject().value(QStringLiteral("gpus")).toInt(), 3);
        QVERIFY(!QJsonDocument(saved).toJson().contains("session-secret"));
        QVERIFY(!manager.isRunning());
        QJsonObject loaded;
        QString loadedPath;
        QString loadError;
        QVERIFY2(manager.loadUserConfig(&loadedPath, &loaded, &loadError), qPrintable(loadError));
        QCOMPARE(loadedPath, profilePath);
        QCOMPARE(loaded.value(QStringLiteral("slurm")).toObject().value(QStringLiteral("host")).toString(), QStringLiteral("test-cluster"));
    }

    void curvePathsDoNotFillBetweenSeries()
    {
        TrainingCurveWidget curve;
        curve.resize(640, 400);
        curve.setMetrics(QJsonObject{{QStringLiteral("steps"), QJsonArray{
            QJsonObject{{QStringLiteral("step"), 1}, {QStringLiteral("training_loss"), 0.5},
                        {QStringLiteral("validation_loss"), 1.0}},
            QJsonObject{{QStringLiteral("step"), 2}, {QStringLiteral("training_loss"), 0.5},
                        {QStringLiteral("validation_loss"), 3.0}},
            QJsonObject{{QStringLiteral("step"), 3}, {QStringLiteral("training_loss"), 0.5},
                        {QStringLiteral("validation_loss"), 1.0}}
        }}});
        curve.show();
        QCoreApplication::processEvents();
        QImage image(curve.size(), QImage::Format_ARGB32_Premultiplied);
        auto renderAndCheck = [&] {
            image.fill(Qt::transparent);
            curve.render(&image);
            QCOMPARE(image.pixelColor(200, 200), QColor(Qt::white));
            bool foundTrainingSeries = false;
            for (int y = 0; y < image.height() && !foundTrainingSeries; ++y) {
                for (int x = 0; x < image.width(); ++x) {
                    if (image.pixelColor(x, y) == QColor(QStringLiteral("#2677b8"))) {
                        foundTrainingSeries = true;
                        break;
                    }
                }
            }
            QVERIFY(foundTrainingSeries);
        };

        renderAndCheck();
        curve.setValidationVisible(false);
        renderAndCheck();
        curve.setValidationVisible(true);
        renderAndCheck();
    }

    void curveSupportsStepsFallbackInvalidAndLogScale()
    {
        TrainingCurveWidget curve;
        curve.resize(720, 360);
        QImage image(curve.size(), QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        curve.render(&image);
        QVERIFY(curve.trainingPoints().isEmpty());
        QVERIFY(!curve.usesStepAxis());
        QVERIFY(curve.xAxisLabel().contains(QStringLiteral("Epoch")));

        curve.setMetrics(QJsonObject{{QStringLiteral("steps"), QJsonArray{
            QJsonObject{{QStringLiteral("step"), 2}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 2.0}, {QStringLiteral("validation_loss"), 1.0}},
            QJsonObject{{QStringLiteral("step"), 4}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 1.0}, {QStringLiteral("validation_loss"), 0.5}},
            QJsonObject{{QStringLiteral("step"), 0}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 1.0}, {QStringLiteral("validation_loss"), 0.5}},
            QJsonObject{{QStringLiteral("step"), 2.5}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 1.0}, {QStringLiteral("validation_loss"), 0.5}},
            QJsonObject{{QStringLiteral("step"), 6}, {QStringLiteral("epoch"), 2},
                        {QStringLiteral("training_loss"), 0.5}, {QStringLiteral("validation_loss"), -0.1}},
            QJsonObject{{QStringLiteral("step"), 7}, {QStringLiteral("epoch"), 2},
                        {QStringLiteral("training_loss"), QStringLiteral("NaN")}, {QStringLiteral("validation_loss"), 0.2}}
        }}});
        QVERIFY(curve.usesStepAxis());
        QCOMPARE(curve.xAxisLabel(), QStringLiteral("Optimizer step"));
        QCOMPARE(curve.trainingPoints().size(), 3);
        QCOMPARE(curve.validationPoints().size(), 3);
        QVERIFY(curve.dataNotice().contains(QStringLiteral("3 invalid")));
        curve.setLogarithmic(true);
        QVERIFY(curve.isLogarithmic());
        QVERIFY(curve.dataNotice().contains(QStringLiteral("non-positive")));
        curve.render(&image);

        curve.setMetrics(QJsonObject{{QStringLiteral("epochs"), QJsonArray{
            QJsonObject{{QStringLiteral("epoch"), 3}, {QStringLiteral("training_loss"), 0.7},
                        {QStringLiteral("validation_loss"), 0.6}}
        }}});
        QVERIFY(!curve.usesStepAxis());
        QCOMPARE(curve.trainingPoints().size(), 1);
        curve.render(&image);
    }

    void submitSnapshotJobsMetricsAndErrors()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        FakeBackend backend;
        QVERIFY(backend.server.listen(QHostAddress::LocalHost));
        NNApplication *application = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
        QVERIFY(application);
        char error[512] = {};
        QVERIFY2(nn_app_create(application, temporary.path().toUtf8().constData(), "ui-test", "UI Test", "0.1.0", false,
                               error, sizeof(error)), error);
        const QByteArray declarations = QByteArrayLiteral("[{\"name\":\"width\",\"type\":\"integer\",\"value\":128}]");
        QVERIFY2(nn_app_set_metavariables_json(application, declarations.constData(), error, sizeof(error)), error);
        QVERIFY2(nn_app_add_node(application, "dense1", "core.linear", "0.1.0", "", 0, 0,
                                 error, sizeof(error)), error);
        QVERIFY2(nn_app_set_parameter_text(application, "dense1", "out_features", "$width",
                                            error, sizeof(error)), error);
        const QByteArray dataset = QByteArrayLiteral("{\"name\":\"Test data\",\"batch\":{\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[1]}},\"targets\":{}}}");
        QVERIFY2(nn_app_create_dataset(application, "test.data", "0.1.0", dataset.constData(), false,
                                       error, sizeof(error)), error);
        const QString projectDirectory = QString::fromUtf8(nn_project_directory(nn_app_project(application)));
        const QString datasetManifestPath = QDir(projectDirectory).filePath(QStringLiteral("datasets/test.data-0.1.0/manifest.json"));
        QFile datasetManifestFile(datasetManifestPath);
        QVERIFY(datasetManifestFile.open(QIODevice::ReadOnly));
        QJsonObject datasetManifest = QJsonDocument::fromJson(datasetManifestFile.readAll()).object();
        datasetManifestFile.close();
        datasetManifest.insert(QStringLiteral("inferenceAssets"), QJsonArray{QStringLiteral("assets/vocab.txt")});
        QVERIFY(datasetManifestFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QCOMPARE(datasetManifestFile.write(QJsonDocument(datasetManifest).toJson(QJsonDocument::Indented)),
                 qint64(QJsonDocument(datasetManifest).toJson(QJsonDocument::Indented).size()));
        datasetManifestFile.close();
        QVERIFY(QDir().mkpath(QDir(projectDirectory).filePath(QStringLiteral("assets"))));
        QFile vocabulary(QDir(projectDirectory).filePath(QStringLiteral("assets/vocab.txt")));
        QVERIFY(vocabulary.open(QIODevice::WriteOnly));
        QCOMPARE(vocabulary.write("digits\n"), qint64(7));
        vocabulary.close();
        QVERIFY(QDir().mkpath(QDir(projectDirectory).filePath(QStringLiteral("data"))));
        QFile sample(QDir(projectDirectory).filePath(QStringLiteral("data/sample.csv")));
        QVERIFY(sample.open(QIODevice::WriteOnly));
        QCOMPARE(sample.write("x,y\n1,2\n"), qint64(8));
        sample.close();

        BackendDialog dialog(application, [application] {
            char saveError[512] = {};
            return nn_app_save(application, saveError, sizeof(saveError));
        }, [](const QString &) { return false; });
        auto *endpoint = dialog.findChild<QLineEdit *>(QStringLiteral("backendEndpoint"));
        endpoint->setText(QStringLiteral("http://127.0.0.1:%1").arg(backend.server.serverPort()));
        dialog.findChild<QLineEdit *>(QStringLiteral("backendToken"))->setText(QStringLiteral("secret-for-test"));
        dialog.show();
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConnect")), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().startsWith(
            QStringLiteral("Connected — podman is ready for training.")) || dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().startsWith(
            QStringLiteral("Backend request completed.")), 3000);
        auto *jobs = dialog.findChild<QTreeWidget *>(QStringLiteral("backendJobs"));
        QTRY_COMPARE_WITH_TIMEOUT(jobs->topLevelItemCount(), 2, 3000);
        const QString existingJobId = jobs->topLevelItem(0)->data(0, Qt::UserRole).toString();
        QVERIFY(jobs->topLevelItem(0)->text(0).contains(QStringLiteral("2026-10-05 12:00")));
        QVERIFY(jobs->topLevelItem(0)->text(0).contains(existingJobId.right(8)));
        QVERIFY(!jobs->topLevelItem(0)->text(0).contains(existingJobId));
        QVERIFY(jobs->topLevelItem(0)->toolTip(0).contains(existingJobId));
        QVERIFY(jobs->topLevelItem(0)->toolTip(0).contains(QStringLiteral("2026-10-05T12:00:00Z")));
        QCOMPARE(jobs->columnCount(), 2);
        jobs->setCurrentItem(jobs->topLevelItem(0));
        auto *metrics = dialog.findChild<QPlainTextEdit *>(QStringLiteral("backendMetrics"));
        QTRY_VERIFY_WITH_TIMEOUT(metrics->toPlainText().contains(QStringLiteral("Final test loss: 0.25")), 3000);
        QVERIFY(metrics->toPlainText().contains(QStringLiteral("Step 1 (epoch 1) — training 0.6, validation 0.5")));
        QVERIFY(metrics->toPlainText().contains(QStringLiteral("Best validation loss: 0.2 (epoch 4)")));
        QVERIFY(metrics->toPlainText().contains(QStringLiteral("Non-improving epochs: 2")));
        QVERIFY(metrics->toPlainText().contains(QStringLiteral("Stopped epoch: 6")));
        auto *curve = dialog.findChild<TrainingCurveWidget *>(QStringLiteral("trainingCurve"));
        QVERIFY(curve->usesStepAxis());
        QCOMPARE(curve->trainingPoints().size(), 120);
        QCOMPARE(dialog.findChild<QLabel *>(QStringLiteral("finalTestLoss"))->text(), QStringLiteral("Final test loss: 0.25"));
        QVERIFY(metrics->verticalScrollBar()->maximum() > 0);
        metrics->verticalScrollBar()->setValue(metrics->verticalScrollBar()->maximum() / 2);
        const int scrollPosition = metrics->verticalScrollBar()->value();
        const int previousDetailCount = backend.detailCount;
        QTRY_VERIFY_WITH_TIMEOUT(backend.detailCount > previousDetailCount, 5000);
        QTest::qWait(100);
        QCOMPARE(metrics->verticalScrollBar()->value(), scrollPosition);

        jobs->setCurrentItem(jobs->topLevelItem(1));
        QTRY_VERIFY_WITH_TIMEOUT(metrics->toPlainText().contains(QStringLiteral("Epoch 3 — training 0.3, validation 0.2")), 3000);
        QVERIFY(!curve->usesStepAxis());
        QCOMPARE(curve->xAxisLabel(), QStringLiteral("Epoch"));
        QCOMPARE(curve->trainingPoints().size(), 1);

        dialog.findChild<QLineEdit *>(QStringLiteral("trainingEpochs"))->setText(QStringLiteral("7"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingBatchSize"))->setText(QStringLiteral("8"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingLearningRate"))->setText(QStringLiteral("0.02"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingSeed"))->setText(QStringLiteral("42"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingPublishEverySteps"))->setText(QStringLiteral("25"));
        dialog.findChild<QComboBox *>(QStringLiteral("trainingDevice"))->setCurrentIndex(1);
        dialog.findChild<QComboBox *>(QStringLiteral("trainingPrecision"))->setCurrentIndex(1);
        dialog.findChild<QComboBox *>(QStringLiteral("trainingOptimizer"))->setCurrentIndex(1);
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingWeightDecay"))->setText(QStringLiteral("0.1"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingGradientClip"))->setText(QStringLiteral("2.5"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingWarmupSteps"))->setText(QStringLiteral("4"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingMaxSteps"))->setText(QStringLiteral("100"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingGradientAccumulationSteps"))->setText(QStringLiteral("8"));
        auto *earlyStoppingPatience = dialog.findChild<QLineEdit *>(QStringLiteral("trainingEarlyStoppingPatience"));
        auto *earlyStoppingMinDelta = dialog.findChild<QLineEdit *>(QStringLiteral("trainingEarlyStoppingMinDelta"));
        QVERIFY(earlyStoppingPatience);
        QVERIFY(earlyStoppingMinDelta);
        QVERIFY(earlyStoppingPatience->toolTip().contains(QStringLiteral("completed epochs")));
        QVERIFY(earlyStoppingMinDelta->toolTip().contains(QStringLiteral("validation loss")));
        earlyStoppingPatience->setText(QStringLiteral("5"));
        earlyStoppingMinDelta->setText(QStringLiteral("nan"));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendSubmit")), Qt::LeftButton);
        QVERIFY(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("finite nonnegative early-stopping settings")));
        QCOMPARE(backend.submitCount, 0);
        earlyStoppingMinDelta->setText(QStringLiteral("0.01"));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendSubmit")), Qt::LeftButton);
        auto *backendStatus = dialog.findChild<QLabel *>(QStringLiteral("backendStatus"));
        QTRY_VERIFY_WITH_TIMEOUT(backend.submitCount == 1 ||
            (backendStatus && !backendStatus->text().contains(QStringLiteral("Submitting"))), 5000);
        QVERIFY2(backend.submitCount == 1, qPrintable(backendStatus->text()));
        const QJsonObject training = backend.submitted.value(QStringLiteral("training")).toObject();
        QCOMPARE(training.value(QStringLiteral("epochs")).toInt(), 7);
        QCOMPARE(training.value(QStringLiteral("batch_size")).toInt(), 8);
        QCOMPARE(training.value(QStringLiteral("learning_rate")).toDouble(), 0.02);
        QCOMPARE(training.value(QStringLiteral("seed")).toInt(), 42);
        QCOMPARE(training.value(QStringLiteral("publish_every_steps")).toInt(), 25);
        QCOMPARE(training.value(QStringLiteral("device")).toString(), QStringLiteral("cuda"));
        QCOMPARE(training.value(QStringLiteral("precision")).toString(), QStringLiteral("bfloat16"));
        QCOMPARE(training.value(QStringLiteral("optimizer")).toString(), QStringLiteral("deepseek-muon"));
        QCOMPARE(training.value(QStringLiteral("weight_decay")).toDouble(), 0.1);
        QCOMPARE(training.value(QStringLiteral("gradient_clip")).toDouble(), 2.5);
        QCOMPARE(training.value(QStringLiteral("warmup_steps")).toInt(), 4);
        QCOMPARE(training.value(QStringLiteral("max_steps")).toInt(), 100);
        QCOMPARE(training.value(QStringLiteral("gradient_accumulation_steps")).toInt(), 8);
        QCOMPARE(training.value(QStringLiteral("early_stopping_patience")).toInt(), 5);
        QCOMPARE(training.value(QStringLiteral("early_stopping_min_delta")).toDouble(), 0.01);
        QCOMPARE(backend.authorization, QByteArrayLiteral("Bearer secret-for-test"));
        const QJsonObject project = backend.submitted.value(QStringLiteral("project")).toObject();
        QVERIFY(project.value(QStringLiteral("manifest")).toObject().value(QStringLiteral("schemaVersion")).toInt() >= 3);
        QVERIFY(project.value(QStringLiteral("nodes")).isArray());
        const QByteArray authoredProject = QJsonDocument(project).toJson(QJsonDocument::Compact);
        QVERIFY(authoredProject.contains("\"$ref\":\"width\""));
        const QJsonObject executionProject = backend.submitted.value(QStringLiteral("execution_project")).toObject();
        QCOMPARE(executionProject.value(QStringLiteral("manifest")).toObject()
                     .value(QStringLiteral("schemaVersion")).toInt(), 2);
        const QByteArray executionBytes = QJsonDocument(executionProject).toJson(QJsonDocument::Compact);
        QVERIFY(executionBytes.contains("\"out_features\":128"));
        QVERIFY(!executionBytes.contains("$ref"));
        const QJsonObject files = backend.submitted.value(QStringLiteral("files")).toObject();
        const QByteArray datasetCode = QByteArray::fromBase64(files.value(QStringLiteral("datasets/test.data-0.1.0/dataset.py")).toString().toLatin1());
        QVERIFY(datasetCode.contains("NotImplementedError"));
        QVERIFY(!QString::fromUtf8(QJsonDocument(backend.submitted).toJson()).contains(QStringLiteral("secret-for-test")));
        QCOMPARE(QByteArray::fromBase64(files.value(QStringLiteral("data/sample.csv")).toString().toLatin1()), QByteArray("x,y\n1,2\n"));
        QCOMPARE(QByteArray::fromBase64(files.value(QStringLiteral("assets/vocab.txt")).toString().toLatin1()), QByteArray("digits\n"));
        QVERIFY(!files.contains(QStringLiteral("data/.DS_Store")));
        QVERIFY(!files.contains(QStringLiteral("model.json")));

        endpoint->setText(QStringLiteral("http://127.0.0.1:%1/bad").arg(backend.server.serverPort()));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConnect")), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("token rejected")), 3000);
        endpoint->setText(QStringLiteral("http://127.0.0.1:%1/unavailable").arg(backend.server.serverPort()));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConnect")), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("training is unavailable")), 3000);
        QVERIFY(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("Container daemon is unavailable.")));
    }
};

QTEST_MAIN(BackendDialogTest)
#include "qt_backend_test.moc"
