#ifndef NN_GUI_MAIN_WINDOW_HPP
#define NN_GUI_MAIN_WINDOW_HPP

#include <QMainWindow>
#include <memory>

class QAction;
class QCheckBox;
class QCloseEvent;
class QLineEdit;
class QLabel;
class QPushButton;
class QTreeWidget;
class QTimer;
class QToolButton;
class QWidget;
struct NNApplication;
struct NNAutomation;
class GraphScene;
class GraphView;
class BackendServiceManager;
enum class FlowDirection;

class MainWindow final : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(NNApplication *application, QWidget *parent = nullptr);
    ~MainWindow() override;

    bool openProject(const QString &directory);
    void showProjectChooser(bool startup = false);
    bool startAutomation(const QString &socketPath);

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void buildUi();
    void refreshAll();
    void refreshPalette();
    void refreshInspector();
    void updateDescriptions();
    void refreshDiagnostics();
    void updateAddNodeButton();
    bool revealNode(const QString &id);
    void updateWindowTitle();
    bool confirmReplaceProject();
    bool saveProject();
    void createProject(const QString &initialTemplate = QString());
    void addSelectedPackage();
    void arrangeCurrentScope(FlowDirection direction);
    void restoreEdit(bool redo);
    void editNodeName(const QString &nodeId, const QString &name);
    void editNodeParameter(const QString &nodeId, const QString &key, const QString &value);
    void createStereotype();
    void createDataset(const QString &id = QString(), const QString &version = QString());
    void importDataset();
    void manageDatasets();
    void manageOperations();
    void manageMetavariables();
    void showParameterCounts();
    static char *automationUiCallback(void *user, const char *operation,
                                     const char *argsJson, char *error, size_t cap);

    std::unique_ptr<NNApplication, void (*)(NNApplication *)> application_;
    GraphScene *scene_ = nullptr;
    GraphView *view_ = nullptr;
    QWidget *paletteTree_ = nullptr;
    QLineEdit *paletteSearch_ = nullptr;
    QTreeWidget *inspector_ = nullptr;
    QLabel *componentGuide_ = nullptr;
    QLabel *selectedNodeDescription_ = nullptr;
    QPushButton *addNodeButton_ = nullptr;
    QWidget *problemsContents_ = nullptr;
    QTreeWidget *diagnostics_ = nullptr;
    QCheckBox *currentScopeProblems_ = nullptr;
    QToolButton *problemsToggle_ = nullptr;
    QToolButton *scopeSelector_ = nullptr;
    QTreeWidget *scopeTree_ = nullptr;
    QAction *undoAction_ = nullptr;
    QAction *redoAction_ = nullptr;
    QTimer *automationTimer_ = nullptr;
    NNAutomation *automation_ = nullptr;
    BackendServiceManager *backendService_ = nullptr;
    QString selectedPaletteId_;
    bool refreshing_ = false;
};

#endif
