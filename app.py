# app.py
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
app.url_map.strict_slashes = False

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False  # Already connected, this edge creates a cycle
        self.parent[root_x] = root_y
        return True

@app.route('/investigate', methods=['POST'])
def investigate():
    try:
        data = request.get_json()
        # Accept either a dict with 'networks' or a list directly
        if isinstance(data, list):
            networks = data
        elif isinstance(data, dict):
            networks = data.get('networks', [])
        else:
            return jsonify({'error': 'Invalid input format'}), 400
        
        result_networks = []
        
        for network_data in networks:
            network_id = network_data['networkId']
            edges = network_data['network']
        
            # Track seen edges and duplicates
            seen = set()
            normalized_edges = []
            duplicates = []
        
            for edge in edges:
                spy1, spy2 = sorted([edge['spy1'], edge['spy2']])
                edge_tuple = (spy1, spy2)
                if edge_tuple not in seen:
                    normalized_edges.append({'spy1': spy1, 'spy2': spy2})
                    seen.add(edge_tuple)
                else:
                    # This is a duplicate edge
                    duplicates.append({'spy1': spy1, 'spy2': spy2})
        
            uf = UnionFind()
            extra_channels = []
        
            for edge in normalized_edges:
                spy1 = edge['spy1']
                spy2 = edge['spy2']
                if not uf.union(spy1, spy2):
                    extra_channels.append(edge)
        
            # Add all duplicates to extra_channels
            extra_channels.extend(duplicates)
        
            result_networks.append({
                'networkId': network_id,
                'extraChannels': extra_channels
            })
        
        result = {'networks': result_networks}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
    app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'
