# Building AI-Powered WAF Log Analysis When Match Data Goes Dark

This guide provides a comprehensive framework for creating an AWS Lambda function with machine learning capabilities to analyze AWS WAF logs, specifically addressing cases where "matched data" fields are blank but requests are still blocked or counted. This is particularly critical for hardware workflow engineers needing to understand WAF rule triggers without explicit match data.

## The challenge: Blind spots in AWS WAF logging

When AWS WAF blocks or counts a request, the matched data field often remains blank, leaving security engineers with insufficient information to identify what triggered the rule. This creates significant blind spots in security monitoring and makes it difficult to verify if rules are working correctly.

The limitations exist because historically, AWS WAF only populated the matched data field for SQL injection and XSS rule statements. Recent updates (February 2024) have extended this to regex-based rules, but many other rule types still result in blank match data. Additional constraints include size restrictions (1,000 byte limit) and partial body inspection capabilities (8KB-64KB depending on the service).

For a hardware workflow engineer building a security monitoring system, these limitations seriously impact visibility into attack patterns and make troubleshooting difficult. Our solution combines advanced log parsing, machine learning, and correlation techniques to overcome these challenges.

## Parsing and analyzing AWS WAF logs from CloudWatch

AWS WAF logs are stored in CloudWatch Logs in JSON format with a rich structure that includes request details even when matched data is blank. To effectively analyze these logs, you need a robust parsing system.

### Lambda-based WAF log parser framework

This Python module provides a foundation for parsing and analyzing WAF logs from CloudWatch:

```python
import boto3
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class WafLogHeader:
    name: str
    value: str

@dataclass
class WafLogHttpRequest:
    clientIp: str
    country: str
    headers: List[WafLogHeader]
    uri: str
    args: str
    httpVersion: str
    httpMethod: str
    requestId: str

@dataclass
class WafLogEntry:
    timestamp: int
    httpRequest: WafLogHttpRequest
    action: str
    terminatingRuleId: Optional[str] = None
    terminatingRuleMatchDetails: Optional[List[Dict]] = None

def lambda_handler(event, context):
    """Lambda handler for AWS WAF log analysis."""
    # Initialize clients outside the handler for connection reuse
    logs_client = boto3.client('logs')
    
    # Configuration
    log_group_name = event.get('logGroupName', '/aws/waf/logs')
    hours_to_analyze = event.get('hoursToAnalyze', 24)
    
    # Calculate time range
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours_to_analyze * 60 * 60 * 1000)
    
    try:
        # Get log streams
        log_streams = get_log_streams(logs_client, log_group_name)
        
        # Process each log stream
        all_logs = []
        for stream in log_streams:
            stream_logs = get_logs_from_stream(
                logs_client,
                log_group_name, 
                stream['logStreamName'],
                start_time,
                end_time
            )
            all_logs.extend(stream_logs)
        
        # Parse logs
        parsed_logs = [parse_waf_log(log) for log in all_logs]
        
        # Filter for logs with blank matched data
        blank_matched_data_logs = [
            log for log in parsed_logs 
            if log.action == "BLOCK" and (
                not log.terminatingRuleMatchDetails or
                not any(detail.get('matchedData') for detail in log.terminatingRuleMatchDetails)
            )
        ]
        
        # Apply machine learning analysis
        analysis_results = analyze_blank_matched_data_logs(blank_matched_data_logs)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'analyzedCount': len(parsed_logs),
                'blankMatchedDataCount': len(blank_matched_data_logs),
                'analysisResults': analysis_results
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def get_log_streams(logs_client, log_group_name: str) -> List[Dict]:
    """Get the available log streams from the log group."""
    response = logs_client.describe_log_streams(
        logGroupName=log_group_name,
        orderBy='LastEventTime',
        descending=True,
        limit=50
    )
    return response['logStreams']

def get_logs_from_stream(logs_client, log_group_name: str, log_stream_name: str, 
                     start_time: int, end_time: int) -> List[Dict]:
    """Retrieve logs from a specific log stream."""
    response = logs_client.get_log_events(
        logGroupName=log_group_name,
        logStreamName=log_stream_name,
        startTime=start_time,
        endTime=end_time,
        limit=10000
    )
    
    # Parse log events
    logs = []
    for event in response['events']:
        try:
            log_data = json.loads(event['message'])
            logs.append(log_data)
        except json.JSONDecodeError:
            print(f"Error parsing log message: {event['message']}")
    
    return logs

def parse_waf_log(log_data: Dict) -> WafLogEntry:
    """Parse a WAF log entry into a structured object."""
    http_request = log_data.get('httpRequest', {})
    
    # Parse headers
    headers = [
        WafLogHeader(header.get("name", ""), header.get("value", ""))
        for header in http_request.get("headers", [])
    ]
    
    # Parse HTTP request
    http_request_obj = WafLogHttpRequest(
        clientIp=http_request.get("clientIp", ""),
        country=http_request.get("country", ""),
        headers=headers,
        uri=http_request.get("uri", ""),
        args=http_request.get("args", ""),
        httpVersion=http_request.get("httpVersion", ""),
        httpMethod=http_request.get("httpMethod", ""),
        requestId=http_request.get("requestId", "")
    )
    
    # Create the log entry
    entry = WafLogEntry(
        timestamp=log_data.get("timestamp", 0),
        httpRequest=http_request_obj,
        action=log_data.get("action", ""),
        terminatingRuleId=log_data.get("terminatingRuleId"),
        terminatingRuleMatchDetails=log_data.get("terminatingRuleMatchDetails")
    )
    
    return entry
```

### Optimizing for large log volumes

WAF logs can accumulate quickly in production environments. To optimize Lambda performance:

1. **Memory configuration**: Allocate appropriate memory based on expected log volume
   - 512MB for small volumes, 1-2GB for medium volumes, 3-4GB for large volumes
   - Use AWS Lambda Power Tuning to find the optimal configuration

2. **Batch processing**: Process logs in batches to improve throughput
   ```python
   def process_logs_in_batches(logs, batch_size=100):
       """Process logs in manageable batches."""
       for i in range(0, len(logs), batch_size):
           batch = logs[i:i+batch_size]
           process_batch(batch)
   ```

3. **Connection reuse**: Initialize clients outside the handler function
   ```python
   # Initialize clients outside the handler for connection reuse
   s3_client = boto3.client('s3')
   cloudwatch_client = boto3.client('logs')
   ```

4. **Asynchronous processing**: Use async functions for IO-bound operations
   ```python
   import asyncio
   
   async def process_log_entry(entry):
       # Process a single log entry
       return result
   
   async def process_logs_parallel(logs):
       tasks = [process_log_entry(entry) for entry in logs]
       return await asyncio.gather(*tasks)
   ```

## Machine learning approaches for identifying patterns in blank matched data

When matched data is missing, machine learning can help identify patterns in the requests that triggered WAF rules. Here are practical ML approaches suited for AWS Lambda implementations in Python.

### Isolation Forest for anomaly detection

Isolation Forest is particularly well-suited for Lambda environments due to its efficiency and low memory requirements:

```python
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

def analyze_blank_matched_data_logs(logs):
    """Analyze logs with blank matched data using Isolation Forest."""
    # Extract features from logs
    features = extract_features_from_logs(logs)
    
    # Convert to DataFrame
    df = pd.DataFrame(features)
    
    # Prepare numerical features
    numeric_cols = ['uri_length', 'query_param_count', 'header_count', 'user_agent_length']
    X_numeric = df[numeric_cols].fillna(0).values
    
    # One-hot encode categorical features
    categorical_cols = ['http_method', 'country', 'rule_id']
    X_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
    
    # Combine features
    X = np.hstack([X_numeric, X_categorical.values])
    
    # Train isolation forest
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.05,
        random_state=42
    )
    
    # Predict anomalies
    df['anomaly_score'] = model.fit_predict(X)
    
    # Identify anomalies (isolation forest returns -1 for anomalies)
    anomalies = df[df['anomaly_score'] == -1]
    
    # Group anomalies by common characteristics
    return analyze_anomaly_patterns(anomalies, logs)

def extract_features_from_logs(logs):
    """Extract features from WAF logs for machine learning analysis."""
    features = []
    
    for log in logs:
        http_req = log.httpRequest
        
        # Extract headers into a dictionary
        headers = {h.name.lower(): h.value for h in http_req.headers}
        
        # Basic request features
        feature = {
            'http_method': http_req.httpMethod,
            'uri': http_req.uri,
            'uri_length': len(http_req.uri),
            'country': http_req.country,
            'rule_id': log.terminatingRuleId,
            
            # Derived features
            'has_query': len(http_req.args) > 0,
            'query_param_count': http_req.args.count('&') + 1 if http_req.args else 0,
            'header_count': len(http_req.headers),
            'user_agent': headers.get('user-agent', ''),
            'user_agent_length': len(headers.get('user-agent', '')),
            'path_depth': http_req.uri.count('/'),
            'contains_special_chars': sum(c in http_req.uri for c in '~!@#$%^&*()=+[]{}\\|;:\'",<>?') > 0,
            'has_numeric_params': any(c.isdigit() for c in http_req.args),
            'path_has_dots': '..' in http_req.uri,
            'has_suspicious_extensions': any(ext in http_req.uri.lower() for ext in ['.php', '.aspx', '.jsp']),
        }
        
        features.append(feature)
    
    return features

def analyze_anomaly_patterns(anomalies, logs):
    """Analyze patterns in detected anomalies."""
    # Group anomalies by rule ID
    rule_groups = anomalies.groupby('rule_id')
    
    patterns = {}
    
    for rule_id, group in rule_groups:
        # Find common features within each rule group
        pattern = {
            'rule_id': rule_id,
            'count': len(group),
            'common_patterns': {}
        }
        
        # Identify common URI patterns
        uri_patterns = extract_common_patterns(group['uri'].tolist())
        if uri_patterns:
            pattern['common_patterns']['uri'] = uri_patterns
        
        # Add other pattern analyses as needed
        patterns[rule_id] = pattern
    
    return patterns

def extract_common_patterns(strings):
    """Extract common substrings or patterns from a list of strings."""
    # This is a simplified implementation
    # More sophisticated pattern extraction would be used in production
    
    # Look for common prefixes
    if not strings:
        return []
    
    common_patterns = []
    
    # Check for common path components
    path_components = {}
    for s in strings:
        parts = s.split('/')
        for part in parts:
            if part:
                path_components[part] = path_components.get(part, 0) + 1
    
    # Find common components that appear in at least 30% of strings
    threshold = max(2, len(strings) * 0.3)
    common_paths = [p for p, count in path_components.items() if count >= threshold]
    
    if common_paths:
        common_patterns.append({
            'type': 'path_component',
            'patterns': common_paths
        })
    
    return common_patterns
```

### Feature engineering for blank matched data

When matched data is blank, focus on these alternative features:

1. **HTTP request components**: Method, URI path, query parameters, headers
2. **Request metadata**: Client IP, country, user agent
3. **Rule information**: The specific rule that was triggered
4. **Derived features**: URI length, path depth, special character presence

A comprehensive feature extraction function:

```python
def extract_waf_features(waf_log):
    """Extract comprehensive features from WAF logs."""
    http_req = waf_log.httpRequest
    
    # Convert headers to dictionary for easier access
    headers = {h.name.lower(): h.value for h in http_req.headers}
    
    # Basic request features
    features = {
        # Request method features
        'method': http_req.httpMethod,
        'is_common_method': http_req.httpMethod in ['GET', 'POST', 'HEAD', 'OPTIONS'],
        
        # URI features
        'uri_length': len(http_req.uri),
        'uri_path_depth': http_req.uri.count('/'),
        'uri_has_extension': '.' in http_req.uri.split('/')[-1],
        'uri_file_extension': http_req.uri.split('.')[-1] if '.' in http_req.uri.split('/')[-1] else '',
        'uri_has_dots': '..' in http_req.uri,
        'uri_special_chars_count': sum(c in http_req.uri for c in '~!@#$%^&*()=+[]{}\\|;:\'",<>?'),
        
        # Query string features
        'has_query': len(http_req.args) > 0,
        'query_length': len(http_req.args),
        'query_param_count': http_req.args.count('&') + 1 if http_req.args else 0,
        'query_has_equals': '=' in http_req.args,
        'query_has_numbers': any(c.isdigit() for c in http_req.args),
        'query_has_special_chars': any(c in '~!@#$%^&*()=+[]{}\\|;:\'",<>?' for c in http_req.args),
        
        # Header features
        'header_count': len(http_req.headers),
        'has_user_agent': 'user-agent' in headers,
        'user_agent_length': len(headers.get('user-agent', '')),
        'has_referer': 'referer' in headers,
        'has_content_type': 'content-type' in headers,
        'content_type': headers.get('content-type', ''),
        'is_json_content': 'json' in headers.get('content-type', '').lower(),
        'is_form_content': 'form' in headers.get('content-type', '').lower(),
        'accept_header_count': sum(1 for h in http_req.headers if h.name.lower().startswith('accept')),
        
        # Client features
        'country': http_req.country,
        'ip_first_octet': int(http_req.clientIp.split('.')[0]) if '.' in http_req.clientIp else 0,
        
        # Rule features
        'rule_id': waf_log.terminatingRuleId,
        'action': waf_log.action,
    }
    
    return features
```

### Incremental learning for continuous improvement

For WAF log analysis, implement an incremental learning approach to continuously update models as new patterns emerge:

```python
from sklearn.linear_model import SGDClassifier
import numpy as np
import joblib
import os

class IncrementalWAFModel:
    """An incremental learning model for WAF logs."""
    
    def __init__(self, model_path='/tmp/waf_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one."""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
            else:
                self.model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
                self.feature_names = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
            self.feature_names = None
    
    def partial_fit(self, X, y, feature_names=None):
        """Update the model with new data."""
        if feature_names is not None:
            self.feature_names = feature_names
        
        classes = np.unique(y)
        
        if not hasattr(self.model, 'classes_'):
            self.model.partial_fit(X, y, classes=classes)
        else:
            self.model.partial_fit(X, y)
        
        self.save_model()
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)
    
    def save_model(self):
        """Save the model to disk."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, self.model_path)
```

## Reverse engineering blocked requests for testing

To properly test and validate WAF rule behavior, you need to reconstruct the original HTTP requests that triggered blocks. This Python module converts WAF logs to curl commands:

```python
import json
import shlex
import urllib.parse
from typing import Dict, List, Optional, Union, Any

class WAFLogToCurl:
    """Convert AWS WAF logs to curl commands for request reproduction."""
    
    def __init__(self, include_client_ip: bool = False):
        """
        Initialize the converter.
        
        Args:
            include_client_ip: Whether to include --interface parameter for specific IPs
        """
        self.include_client_ip = include_client_ip
    
    def parse_log_entry(self, log_data: Union[str, Dict]) -> Dict:
        """
        Parse a WAF log entry (either a JSON string or a dictionary).
        
        Args:
            log_data: WAF log entry, either a JSON string or a dictionary
            
        Returns:
            Dict: Parsed log data
        """
        if isinstance(log_data, str):
            try:
                return json.loads(log_data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in WAF log entry")
        return log_data
    
    def extract_request_info(self, log_entry: Dict) -> Dict[str, Any]:
        """
        Extract relevant request information from a WAF log entry.
        
        Args:
            log_entry: Parsed WAF log entry
            
        Returns:
            Dict: Extracted request information
        """
        if "httpRequest" not in log_entry:
            raise ValueError("Missing httpRequest in WAF log entry")
        
        http_request = log_entry["httpRequest"]
        
        # Basic request info
        info = {
            "method": http_request.get("httpMethod", "GET"),
            "uri": http_request.get("uri", "/"),
            "headers": {},
            "client_ip": http_request.get("clientIp"),
            "args": http_request.get("args", "")
        }
        
        # Process headers
        for header in http_request.get("headers", []):
            name = header.get("name")
            value = header.get("value")
            if name and value:
                info["headers"][name] = value
        
        # Build full URL
        host = info["headers"].get("Host", "")
        url = f"https://{host}{info['uri']}"
        
        # Add query parameters if present
        if info["args"]:
            url = f"{url}?{info['args']}"
        
        info["url"] = url
        
        # Try to extract body from matched data if present
        body = None
        if "terminatingRuleMatchDetails" in log_entry:
            for match_detail in log_entry["terminatingRuleMatchDetails"]:
                if match_detail.get("location") == "BODY":
                    body_data = match_detail.get("matchedData", [])
                    if body_data:
                        body = " ".join(body_data)
        
        info["body"] = body
        
        return info
    
    def to_curl(self, log_entry: Union[str, Dict], include_verbose: bool = False) -> str:
        """
        Convert a WAF log entry to a curl command.
        
        Args:
            log_entry: WAF log entry, either JSON string or dictionary
            include_verbose: Whether to include -v (verbose) flag
            
        Returns:
            str: curl command that reproduces the request
        """
        # Parse and extract request info
        parsed_entry = self.parse_log_entry(log_entry)
        request_info = self.extract_request_info(parsed_entry)
        
        # Start building curl command
        curl_parts = ["curl"]
        
        # Add verbose flag if requested
        if include_verbose:
            curl_parts.append("-v")
        
        # Add method flag if not GET
        if request_info["method"] != "GET":
            curl_parts.extend(["-X", request_info["method"]])
        
        # Add headers
        for name, value in request_info["headers"].items():
            # Skip Host header as it's included in the URL
            if name.lower() == "host":
                continue
            curl_parts.extend(["-H", f"{name}: {value}"])
        
        # Add body if present
        if request_info.get("body"):
            curl_parts.extend(["-d", request_info["body"]])
        
        # Add client IP if requested
        if self.include_client_ip and request_info.get("client_ip"):
            curl_parts.extend(["--interface", request_info["client_ip"]])
        
        # Add URL
        curl_parts.append(f"'{request_info['url']}'")
        
        return " ".join(curl_parts)
```

### Handling incomplete log data

When dealing with incomplete WAF logs (especially when matched data is blank), use this enhanced processor:

```python
def reconstruct_request_from_incomplete_data(log_entry):
    """Attempt to reconstruct missing request information."""
    http_request = log_entry.get("httpRequest", {})
    
    # If matched data fields are blank but we know the rule type
    if "terminatingRuleMatchDetails" in log_entry and not log_entry["terminatingRuleMatchDetails"]:
        rule_id = log_entry.get("terminatingRuleId", "")
        
        # Try to infer the type of attack based on rule ID
        if "SQLi" in rule_id:
            log_entry["terminatingRuleMatchDetails"] = [{
                "conditionType": "SQL_INJECTION",
                "location": "UNKNOWN",
                "matchedData": []
            }]
        elif "XSS" in rule_id:
            log_entry["terminatingRuleMatchDetails"] = [{
                "conditionType": "XSS",
                "location": "UNKNOWN",
                "matchedData": []
            }]
        elif "BadBot" in rule_id or "Bot" in rule_id:
            log_entry["terminatingRuleMatchDetails"] = [{
                "conditionType": "BOT_CONTROL",
                "location": "HEADER",
                "matchedData": []
            }]
    
    # Ensure headers exist
    if "headers" not in http_request:
        http_request["headers"] = []
        
    # Add required headers if missing
    host_exists = any(h.get("name") == "Host" for h in http_request.get("headers", []))
    if not host_exists and "httpSourceName" in log_entry:
        source_name = log_entry["httpSourceName"]
        if source_name in ["ALB", "APIGW", "CF"]:
            http_request["headers"].append({
                "name": "Host",
                "value": log_entry.get("httpSourceId", "example.com")
            })
    
    # Ensure method exists
    if "httpMethod" not in http_request:
        http_request["httpMethod"] = "GET"
        
    # Update the log entry
    log_entry["httpRequest"] = http_request
    
    return log_entry
```

## Real-time analysis and alerting solutions

Real-time analysis of WAF logs enables immediate response to security threats. Here's how to implement real-time monitoring and alerting in Lambda.

### Streaming architecture for WAF logs

For real-time analysis, use this streaming architecture:

```
[WAF] → [Kinesis Data Firehose] → [Lambda (Analysis)] → [SNS/EventBridge]
                                 → [S3 (Archive)]
```

The Lambda function implementation:

```python
import json
import boto3
import base64
import time
from typing import Dict, List, Any

def lambda_handler(event, context):
    """Process WAF logs in real-time for alerting."""
    sns_client = boto3.client('sns')
    events_client = boto3.client('events')
    dynamodb = boto3.resource('dynamodb')
    state_table = dynamodb.Table('waf-analysis-state')
    
    alerts = []
    
    # Process records from Kinesis Data Firehose
    for record in event['Records']:
        # Decode and parse the record
        payload = base64.b64decode(record['data']).decode('utf-8')
        try:
            log_entry = json.loads(payload)
            
            # Process only if action is BLOCK
            if log_entry.get('action') == 'BLOCK':
                # Check if matched data is blank
                has_matched_data = False
                if 'terminatingRuleMatchDetails' in log_entry:
                    for detail in log_entry['terminatingRuleMatchDetails']:
                        if detail.get('matchedData'):
                            has_matched_data = True
                            break
                
                if not has_matched_data:
                    # Extract key information
                    client_ip = log_entry.get('httpRequest', {}).get('clientIp', '')
                    uri = log_entry.get('httpRequest', {}).get('uri', '')
                    rule_id = log_entry.get('terminatingRuleId', '')
                    
                    # Update state for this IP
                    state = update_ip_state(state_table, client_ip, rule_id, uri)
                    
                    # Check for alert conditions
                    if should_alert(state, log_entry):
                        alert = create_alert(log_entry, state)
                        alerts.append(alert)
                        
                        # Send alert to SNS
                        sns_client.publish(
                            TopicArn='arn:aws:sns:region:account-id:waf-alerts',
                            Message=json.dumps(alert),
                            Subject=f'WAF Alert: {alert["alertType"]}'
                        )
                        
                        # Send event to EventBridge for automation
                        events_client.put_events(
                            Entries=[{
                                'Source': 'custom.waf.monitoring',
                                'DetailType': 'WAF Security Alert',
                                'Detail': json.dumps(alert),
                                'EventBusName': 'default'
                            }]
                        )
        except json.JSONDecodeError as e:
            print(f"Error parsing record: {e}")
        except Exception as e:
            print(f"Error processing record: {e}")
    
    return {
        'statusCode': 200,
        'processedRecords': len(event['Records']),
        'alertsGenerated': len(alerts)
    }

def update_ip_state(table, ip_address, rule_id, uri):
    """Update and retrieve state information for an IP address."""
    timestamp = int(time.time())
    
    try:
        response = table.update_item(
            Key={'ip': ip_address},
            UpdateExpression='SET #ts = :ts, requestCount = if_not_exists(requestCount, :start) + :inc, lastUri = :uri, lastRuleId = :rule',
            ExpressionAttributeNames={'#ts': 'lastSeen'},
            ExpressionAttributeValues={
                ':ts': timestamp,
                ':start': 0,
                ':inc': 1,
                ':uri': uri,
                ':rule': rule_id
            },
            ReturnValues='ALL_NEW'
        )
        return response.get('Attributes', {})
    except Exception as e:
        print(f"Error updating state: {e}")
        return {'ip': ip_address, 'requestCount': 1, 'lastSeen': timestamp, 'lastUri': uri, 'lastRuleId': rule_id}

def should_alert(state, log_entry):
    """Determine if an alert should be generated based on state and current log."""
    # Alert on high request count
    if state.get('requestCount', 0) >= 10:
        return True
    
    # Alert on certain rule IDs regardless of count
    critical_rules = ['SQLi', 'XSS', 'RFI', 'LFI', 'RCE']
    rule_id = log_entry.get('terminatingRuleId', '')
    if any(rule in rule_id for rule in critical_rules):
        return True
    
    # More complex alert logic can be added here
    
    return False

def create_alert(log_entry, state):
    """Create a structured alert from a log entry and state."""
    http_request = log_entry.get('httpRequest', {})
    
    alert = {
        'timestamp': int(time.time()),
        'alertType': determine_alert_type(log_entry),
        'clientIp': http_request.get('clientIp', ''),
        'country': http_request.get('country', ''),
        'uri': http_request.get('uri', ''),
        'method': http_request.get('httpMethod', ''),
        'ruleId': log_entry.get('terminatingRuleId', ''),
        'requestCount': state.get('requestCount', 1),
        'firstSeen': state.get('firstSeen', int(time.time())),
        'lastSeen': state.get('lastSeen', int(time.time())),
        'severity': determine_severity(log_entry, state)
    }
    
    return alert

def determine_alert_type(log_entry):
    """Determine the type of alert based on the rule ID."""
    rule_id = log_entry.get('terminatingRuleId', '')
    
    if 'SQLi' in rule_id:
        return 'SQL Injection Attempt'
    elif 'XSS' in rule_id:
        return 'Cross-Site Scripting Attempt'
    elif 'Bot' in rule_id or 'BadBot' in rule_id:
        return 'Bot Activity'
    elif 'RateLimit' in rule_id or 'RateBase' in rule_id:
        return 'Rate Limit Exceeded'
    else:
        return 'WAF Rule Triggered'

def determine_severity(log_entry, state):
    """Determine the severity of an alert."""
    rule_id = log_entry.get('terminatingRuleId', '')
    request_count = state.get('requestCount', 1)
    
    # Critical severity for certain attack types
    if any(rule in rule_id for rule in ['SQLi', 'XSS', 'RFI', 'LFI', 'RCE']):
        return 'CRITICAL'
    
    # High severity for repeated blocks
    if request_count >= 20:
        return 'HIGH'
    elif request_count >= 10:
        return 'MEDIUM'
    else:
        return 'LOW'
```

### Correlation with other security signals

To improve alert accuracy, correlate WAF logs with other security signals:

```python
def correlate_security_signals(waf_log, context):
    """Correlate WAF logs with other security signals."""
    client_ip = waf_log.get('httpRequest', {}).get('clientIp', '')
    
    # Check GuardDuty findings
    guardduty_findings = query_guardduty_findings(client_ip)
    
    # Check CloudTrail for suspicious activity
    cloudtrail_events = query_cloudtrail_events(client_ip)
    
    # Check VPC Flow Logs
    flow_log_findings = query_flow_logs(client_ip)
    
    # Calculate risk score based on correlated signals
    risk_score = calculate_risk_score(
        waf_log, 
        guardduty_findings, 
        cloudtrail_events, 
        flow_log_findings
    )
    
    return {
        'waf_log': waf_log,
        'correlated_signals': {
            'guardduty': guardduty_findings,
            'cloudtrail': cloudtrail_events,
            'flow_logs': flow_log_findings
        },
        'risk_score': risk_score
    }

def query_guardduty_findings(ip_address):
    """Query GuardDuty findings for an IP address."""
    guardduty = boto3.client('guardduty')
    
    # Get detector ID
    detectors = guardduty.list_detectors()
    if not detectors['DetectorIds']:
        return []
    
    detector_id = detectors['DetectorIds'][0]
    
    # Query findings for the IP
    response = guardduty.list_findings(
        DetectorId=detector_id,
        FindingCriteria={
            'Criterion': {
                'resource.instanceDetails.networkInterfaces.privateIpAddresses.privateIpAddress': {
                    'Eq': [ip_address]
                }
            }
        },
        MaxResults=10
    )
    
    # Get finding details
    if response['FindingIds']:
        findings = guardduty.get_findings(
            DetectorId=detector_id,
            FindingIds=response['FindingIds']
        )
        return findings['Findings']
    
    return []
```

## Workarounds for AWS WAF logging limitations

There are several approaches to overcome the limitations in AWS WAF logging, particularly when matched data fields are blank.

### Converting rules to regex for better visibility

Since February 2024, AWS WAF supports matched data for regex rules. Convert string match rules to regex patterns to get better visibility:

```python
def convert_string_match_to_regex(rule_group_name, rule_name, string_match):
    """
    Convert a string match rule to a regex rule for better logging.
    This function generates the necessary AWS CLI commands.
    
    Args:
        rule_group_name: The name of the rule group
        rule_name: The name of the rule
        string_match: The string to match
    
    Returns:
        str: AWS CLI command to update the rule
    """
    # Escape special regex characters in the string match
    escaped_match = re.escape(string_match)
    
    # Generate the AWS CLI command
    command = f"""aws wafv2 update-rule-group \\
    --name {rule_group_name} \\
    --scope REGIONAL \\
    --id YOUR_RULE_GROUP_ID \\
    --rules '[{{
        "Name": "{rule_name}",
        "Priority": 0,
        "Statement": {{
            "RegexMatchStatement": {{
                "RegexString": "{escaped_match}",
                "FieldToMatch": {{
                    "UriPath": {{}}
                }},
                "TextTransformations": [
                    {{
                        "Priority": 0,
                        "Type": "NONE"
                    }}
                ]
            }}
        }},
        "Action": {{
            "Block": {{}}
        }},
        "VisibilityConfig": {{
            "SampledRequestsEnabled": true,
            "CloudWatchMetricsEnabled": true,
            "MetricName": "{rule_name}"
        }}
    }}]'"""
    
    return command
```

### Using GetSampledRequests for additional insights

When matched data is blank, use the GetSampledRequests API to get additional information:

```python
def get_sampled_requests(web_acl_id, rule_id, time_window=120):
    """
    Get sampled requests for a specific rule to gain additional insights.
    
    Args:
        web_acl_id: The WebACL ID
        rule_id: The rule ID to sample
        time_window: Time window in minutes (default: 120)
    
    Returns:
        List: Sampled requests
    """
    wafv2_client = boto3.client('wafv2')
    
    # Calculate time window
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=time_window)
    
    # Get sampled requests
    response = wafv2_client.get_sampled_requests(
        WebAclArn=web_acl_id,
        RuleMetricName=rule_id,
        Scope='REGIONAL',  # or 'CLOUDFRONT'
        TimeWindow={
            'StartTime': start_time,
            'EndTime': end_time
        },
        MaxItems=100
    )
    
    return response.get('SampledRequests', [])
```

### Dual-mode rules for enhanced visibility

Create shadow rules in COUNT mode to gather more information about what's being blocked:

```python
def create_shadow_rule(web_acl_id, original_rule):
    """
    Create a shadow rule in COUNT mode to gather additional information.
    
    Args:
        web_acl_id: The WebACL ID
        original_rule: The original rule configuration
    
    Returns:
        Dict: Response from AWS API
    """
    wafv2_client = boto3.client('wafv2')
    
    # Create a copy of the rule with a different name and COUNT action
    shadow_rule = original_rule.copy()
    shadow_rule['Name'] = f"{original_rule['Name']}_Shadow"
    shadow_rule['Action'] = {'Count': {}}
    
    # If the original rule is not using RegexMatchStatement, convert it
    if 'RegexMatchStatement' not in str(shadow_rule['Statement']):
        shadow_rule = convert_to_regex_rule(shadow_rule)
    
    # Get current rules
    response = wafv2_client.get_web_acl(
        Name=web_acl_id.split('/')[-1],
        Scope='REGIONAL',
        Id=web_acl_id.split('/')[-2]
    )
    
    current_rules = response['WebACL']['Rules']
    current_rules.append(shadow_rule)
    
    # Update web ACL with the new rule
    update_response = wafv2_client.update_web_acl(
        Name=web_acl_id.split('/')[-1],
        Scope='REGIONAL',
        Id=web_acl_id.split('/')[-2],
        DefaultAction=response['WebACL']['DefaultAction'],
        Rules=current_rules,
        VisibilityConfig=response['WebACL']['VisibilityConfig'],
        LockToken=response['LockToken']
    )
    
    return update_response
```

## Existing tools and solutions for WAF log analysis

Several existing tools can help analyze AWS WAF logs, especially when matched data fields are blank.

### AWS Native Solutions

1. **CloudWatch Logs Insights** - Powerful query interface for analyzing WAF logs
   ```sql
   fields @timestamp, httpRequest.clientIp, terminatingRuleId, action
   | filter action = "BLOCK" and ispresent(terminatingRuleMatchDetails) = false
   | sort @timestamp desc
   | limit 100
   ```

2. **Amazon Athena** - SQL-based querying for large-scale WAF log analysis
   ```sql
   SELECT timestamp, httpRequest.clientIp, terminatingRuleId, action 
   FROM waf_logs 
   WHERE action = 'BLOCK' 
     AND terminatingRuleMatchDetails IS NULL 
     AND date >= date_format(current_date - interval '7' day, '%Y/%m/%d')
   ORDER BY timestamp DESC
   LIMIT 100;
   ```

3. **Amazon OpenSearch Service** - Powerful search and visualization for security logs

### Open-Source Tools

1. **CloudWatch Dashboard for AWS WAF** - Ready-made dashboards for WAF analysis
   - GitHub: [CloudWatch Dashboard for AWS WAF](https://github.com/ytkoka/CloudWatch-Dashboard-for-AWS-WAF)
   - Features: Pre-built dashboards, contributor insights rules

2. **AWS WAF Operations Dashboards** - OpenSearch-based dashboards for WAF
   - GitHub: [AWS WAF Operations Dashboards](https://github.com/aws-samples/aws-waf-ops-dashboards)
   - Features: Visualizations for WAF traffic and security events

3. **AWS WAF Logger** - Lambda-based enhanced logging for WAF
   - GitHub: [mybuilder/aws-waf-logger](https://github.com/mybuilder/aws-waf-logger)

### Integration with your Lambda solution

To incorporate these existing tools with your custom Lambda function:

```python
def lambda_handler(event, context):
    """Main handler that coordinates with existing tools."""
    # Process logs with custom code
    processed_logs = process_waf_logs(event)
    
    # Output to formats compatible with existing tools
    output_to_opensearch(processed_logs)
    output_to_athena_compatible_format(processed_logs)
    
    # Generate alerts based on analysis
    alerts = generate_alerts(processed_logs)
    
    return {
        'statusCode': 200,
        'processedLogs': len(processed_logs),
        'alertsGenerated': len(alerts)
    }

def output_to_opensearch(logs):
    """Format and send logs to OpenSearch for use with dashboards."""
    opensearch = boto3.client('opensearch')
    # Implementation details...

def output_to_athena_compatible_format(logs):
    """Save logs in a format optimized for Athena queries."""
    s3_client = boto3.client('s3')
    # Implementation details...
```

## Conclusion

This guide provides a comprehensive framework for building an AWS Lambda function with machine learning capabilities to analyze AWS WAF logs, particularly focusing on cases where "matched data" fields are blank. By combining robust log parsing, ML-based pattern detection, request reconstruction, real-time analysis, and an understanding of WAF logging limitations, hardware workflow engineers can implement powerful solutions to improve security visibility.

The key to successfully analyzing requests with blank matched data is to leverage multiple approaches in combination: use regex rules where possible, implement ML-based pattern detection, correlate across security signals, and integrate with existing analysis tools. This multi-faceted approach ensures maximum visibility into WAF-blocked requests, enabling better security monitoring and rule optimization.