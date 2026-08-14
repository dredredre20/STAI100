# Running the Tests in Terminal

Since we have multiple tests involved — unit, trajectory, and end-to-end — there are several ways to run the each by component. When running a tests, make sure your path is inside the main `evaluation` directory. You can replace the `py -m` part of the runtime with your specific python version in the desktop.

## Unit Tests
### 1. Run All Tests in Unit Tests:
```bash
py -m pytest unit_tests\ -v -s
```

### 2. Run a specific Test
```bash
py -m pytest unit_tests\\test_job_retrieval.py -v -s
```

---

## Trajectory Tests
### 1. Run All Tests in Trajectory Tests:
```bash
py -m pytest trajectory_tests\ -v -s
```

### 2. Run a specific Test
```bash
py -m pytest trajectory_tests\\test_tool_selection.py -v -s
```
---

## End-to-End Tests

### 1. Run All Tests in E2E:
```bash
py -m pytest e2e_tests\ -v -s
```

### 2. Run a specific Test
```bash
py -m pytest e2e_tests\\test_agentic_llm.py -v -s
```
