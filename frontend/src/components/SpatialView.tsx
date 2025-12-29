type Asset = {
    id: string;
    x: number;
    y: number;
    type: string;
};

const SpatialView = ({
    assets = []
}: {
    assets?: Asset[]
}) => {
    const size = 300;
    const center = size / 2;
    // Scale: 300px width represeting 200m dia (100m radius) -> 1.5 px/m
    const scale = 1.5;

    // console.log("Rendering SpatialView", assets);

    const gridRings = [50, 100];

    return (
        <div className="spatial-card">
            <div className="card-header">
                <h2>Spatial Awareness</h2>
                <span className="pill">Relative</span>
            </div>
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                {/* Grid Rings */}
                {gridRings.map(r => (
                    <circle
                        key={r}
                        cx={center}
                        cy={center}
                        r={r * scale}
                        className="radar-grid"
                        fill="none"
                    />
                ))}

                {/* Axis */}
                <line x1={0} y1={center} x2={size} y2={center} className="radar-grid" />
                <line x1={center} y1={0} x2={center} y2={size} className="radar-grid" />

                {/* Assets */}
                {assets.map(asset => (
                    <circle
                        key={asset.id}
                        cx={center + (asset.x * scale)}
                        cy={center - (asset.y * scale)}
                        r="4"
                        className="radar-asset"
                    />
                ))}

                {/* Vehicle (Center) */}
                <path
                    d="M-6 4 L0 -8 L6 4 Z"
                    transform={`translate(${center}, ${center})`}
                    className="radar-vessel"
                />
            </svg>
            <p className="note">Scanning 50m radius</p>
        </div>
    );
};

export default SpatialView;
